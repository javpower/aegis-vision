//! `.avpack` 数据集打包容器（PLAN 附录 B 格式 v1）。
//!
//! 解决"海量小文件 IO"：把数据集合并为单文件容器 + blake3 索引，训练时
//! memmap 直读（[`AvPackReader`] 持有只读映射：容器不进堆、条目字节零拷贝
//! 切片、按名查找 O(1)）。布局：
//!
//! ```text
//! 0x00      magic "AVPK"（4B）
//! 0x04      version u32 LE = 1
//! 0x08      flags u32 LE（bit0 预留 zstd；v1 恒 0）
//! 0x0C      index_offset u64 LE（索引区绝对偏移）
//! 0x14      index_len u64 LE
//! 0x1C      index_blake3 32B（覆盖索引区字节）
//! 0x3C..0x40 预留
//! 数据区    各文件原样字节依次排列（entry.offset 相对数据区起点）
//! 索引区    count u32 + entry×N：
//!           name_len u32 LE + name(UTF-8) + blake3 32B + offset u64 + len u64
//! ```
//!
//! 注：PLAN 原稿的尾部 CRC32 以 blake3 替代（更强，无新依赖）；zstd 压缩留 flags 位。

use std::collections::{BTreeMap, HashMap};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use blake3::Hasher;

use av_core::error::{AvError, AvResult};

const MAGIC: &[u8; 4] = b"AVPK";
const VERSION: u32 = 1;
const HEADER_LEN: u64 = 0x40;

/// 容器内一个条目的元信息。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub name: String,
    pub hash: [u8; 32],
    pub offset: u64,
    pub len: u64,
}

/// 递归收集目录下全部文件（相对路径，`\` 统一为 `/`，排序保证确定性）。
pub fn collect_files(root: &Path) -> AvResult<Vec<PathBuf>> {
    fn walk(out: &mut Vec<PathBuf>, root: &Path, dir: &Path) -> AvResult<()> {
        for e in std::fs::read_dir(dir)? {
            let p = e?.path();
            if p.is_dir() {
                walk(out, root, &p)?;
            } else {
                let rel = p
                    .strip_prefix(root)
                    .map_err(|_| AvError::data("相对路径计算失败"))?
                    .to_string_lossy()
                    .replace('\\', "/");
                out.push(PathBuf::from(rel));
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    walk(&mut out, root, root)?;
    out.sort();
    if out.is_empty() {
        return Err(AvError::data(format!(
            "打包源目录为空（无任何文件）: {}",
            root.display()
        )));
    }
    Ok(out)
}

/// 把 `src` 目录流式打包为 `.avpack` 单文件容器。返回 (条目数, 数据区总字节)。
pub fn pack_dir(src: &Path, out_path: &Path) -> AvResult<(usize, u64)> {
    let files = collect_files(src)?;
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::fs::File::create(out_path)?;
    // 头部占位（最后回填）
    f.write_all(&[0u8; HEADER_LEN as usize])?;

    let mut entries: Vec<Entry> = Vec::new();
    let mut data_len: u64 = 0;
    for rel in &files {
        let bytes = std::fs::read(src.join(rel))?;
        let mut h = Hasher::new();
        h.update(&bytes);
        let hash: [u8; 32] = h.finalize().into();
        f.write_all(&bytes)?;
        entries.push(Entry {
            name: rel.to_string_lossy().replace('\\', "/"),
            hash,
            offset: data_len,
            len: bytes.len() as u64,
        });
        data_len += bytes.len() as u64;
    }

    // 索引区（名字序；构建时 entries 已按 collect_files 的排序，BTreeMap 语义等价）
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    let mut index: Vec<u8> = Vec::new();
    index.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for e in &entries {
        index.extend_from_slice(&(e.name.len() as u32).to_le_bytes());
        index.extend_from_slice(e.name.as_bytes());
        index.extend_from_slice(&e.hash);
        index.extend_from_slice(&e.offset.to_le_bytes());
        index.extend_from_slice(&e.len.to_le_bytes());
    }
    let mut ih = Hasher::new();
    ih.update(&index);
    let index_hash: [u8; 32] = ih.finalize().into();
    f.write_all(&index)?;

    // 回填头部
    let index_offset = data_len + HEADER_LEN;
    f.seek(SeekFrom::Start(0))?;
    f.write_all(MAGIC)?;
    f.seek(SeekFrom::Start(4))?;
    f.write_all(&VERSION.to_le_bytes())?;
    f.seek(SeekFrom::Start(8))?;
    f.write_all(&0u32.to_le_bytes())?;
    f.seek(SeekFrom::Start(12))?;
    f.write_all(&index_offset.to_le_bytes())?;
    f.seek(SeekFrom::Start(20))?;
    f.write_all(&(index.len() as u64).to_le_bytes())?;
    f.seek(SeekFrom::Start(28))?;
    f.write_all(&index_hash)?;
    f.sync_all()?;
    Ok((entries.len(), data_len))
}

/// 已打开容器的只读句柄：打开时校验 magic/版本/索引哈希。
///
/// 数据区经 memmap 直读（模块文档的「训练时 memmap 直读」由此实现）：容器
/// 字节不进堆，条目读取 = mmap 切片零拷贝；条目名 → 下标哈希表在打开时
/// 建好，按名查找 O(1)（旧实现整容器 `fs::read` 进堆 + 每样本线性扫条目表）。
pub struct AvPackReader {
    map: memmap2::Mmap,
    data_start: u64,
    entries: Vec<Entry>,
    by_name: HashMap<String, usize>,
}

/// 建只读映射。Windows 下对共享冲突/字节锁做小退避重试：刚写完的容器紧接
/// mmap 会被杀软/索引器的瞬时句柄绊倒（"打包 → 立即训练"实测偶发，冲突只
/// 持续毫秒级）；其余错误码原样返回，非 Windows 单次尝试。
#[cfg(windows)]
fn open_mmap(f: &std::fs::File) -> std::io::Result<memmap2::Mmap> {
    const SHARING_VIOLATION: i32 = 32; // ERROR_SHARING_VIOLATION
    const LOCK_VIOLATION: i32 = 33; // ERROR_LOCK_VIOLATION
    let mut last = None;
    for attempt in 0..4 {
        if attempt > 0 {
            std::thread::sleep(std::time::Duration::from_millis(50 * attempt as u64));
        }
        // SAFETY: 只读映射，见 open() 处注释
        match unsafe { memmap2::Mmap::map(f) } {
            Ok(m) => return Ok(m),
            Err(e) if matches!(e.raw_os_error(), Some(code) if code == SHARING_VIOLATION || code == LOCK_VIOLATION) => {
                last = Some(e)
            }
            Err(e) => return Err(e),
        }
    }
    Err(last.expect("重试耗尽时必已有错误"))
}

#[cfg(not(windows))]
fn open_mmap(f: &std::fs::File) -> std::io::Result<memmap2::Mmap> {
    // SAFETY: 只读映射，见 open() 处注释
    unsafe { memmap2::Mmap::map(f) }
}

impl AvPackReader {
    pub fn open(path: &Path) -> AvResult<Self> {
        let f = std::fs::File::open(path)?;
        let file_len = f.metadata()?.len() as usize;
        if file_len < HEADER_LEN as usize {
            return Err(AvError::data(format!(
                "{} 不是合法的 avpack 容器（文件头不完整）",
                path.display()
            )));
        }
        // SAFETY: 只读映射；容器按不可变数据集约定使用（打包后不再写入，
        // 并发截断属未定义使用，越界读取由下方 len 检查 + 条目边界检查拦截）
        let map = open_mmap(&f)?;
        let data = &map[..];
        if &data[0..4] != MAGIC {
            return Err(AvError::data(format!(
                "{} 不是合法的 avpack 容器（magic 不符）",
                path.display()
            )));
        }
        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if version != VERSION {
            return Err(AvError::data(format!(
                "avpack 版本不支持: {version}（当前支持 {VERSION}）"
            )));
        }
        let index_offset = u64::from_le_bytes(data[12..20].try_into().unwrap()) as usize;
        let index_len = u64::from_le_bytes(data[20..28].try_into().unwrap()) as usize;
        let expect_hash: [u8; 32] = data[28..60].try_into().unwrap();
        if index_offset + index_len > data.len() {
            return Err(AvError::data("avpack 索引区越界"));
        }
        let index_bytes = &data[index_offset..index_offset + index_len];
        let mut ih = Hasher::new();
        ih.update(index_bytes);
        if ih.finalize().as_bytes() != &expect_hash {
            return Err(AvError::data("avpack 索引哈希校验失败（文件损坏或被篡改）"));
        }
        let mut entries = Vec::new();
        // 跳过 count 字段（前 4 字节），从第一个 entry 开始解析——
        // 此前的 p=0 起步会把 count 当 nl 读，整体错位 4 字节（真 bug 实录）
        let mut p = 4usize;
        let count = u32::from_le_bytes(index_bytes[0..4].try_into().unwrap()) as usize;
        // 索引哈希只保证哈希一致，挡不住截断/恶意构造的 count 与 name 长度；
        // release 推理 profile 是 panic=abort，切片越界会直接崩进程——逐字段
        // 带边界检查，越界一律报数据错误
        let rd_u64 = |p: usize| -> AvResult<u64> {
            index_bytes
                .get(p..p + 8)
                .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
                .ok_or_else(|| AvError::data("avpack 索引区越界（条目字段截断）"))
        };
        for _ in 0..count {
            let nl = index_bytes
                .get(p..p + 4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)
                .ok_or_else(|| AvError::data("avpack 索引区越界（条目名长度截断）"))?;
            p += 4;
            let name = String::from_utf8(
                index_bytes
                    .get(p..p + nl)
                    .ok_or_else(|| AvError::data("avpack 索引区越界（条目名截断）"))?
                    .to_vec(),
            )
            .map_err(|_| AvError::data("索引名字非 UTF-8"))?;
            p += nl;
            let hash: [u8; 32] = index_bytes
                .get(p..p + 32)
                .ok_or_else(|| AvError::data("avpack 索引区越界（条目哈希截断）"))?
                .try_into()
                .unwrap();
            p += 32;
            let offset = rd_u64(p)?;
            p += 8;
            let len = rd_u64(p)?;
            p += 8;
            entries.push(Entry {
                name,
                hash,
                offset,
                len,
            });
        }
        let mut by_name = HashMap::with_capacity(entries.len());
        for (i, e) in entries.iter().enumerate() {
            by_name.insert(e.name.clone(), i);
        }
        Ok(Self {
            map,
            data_start: HEADER_LEN,
            entries,
            by_name,
        })
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// 按名查条目元信息（O(1)；旧实现每样本线性扫条目表）。
    pub fn entry(&self, name: &str) -> Option<&Entry> {
        self.by_name.get(name).map(|&i| &self.entries[i])
    }

    /// 按名读条目字节（零拷贝借用，mmap 切片）。
    ///
    /// 不做条目级 blake3 校验——打开时索引哈希已校验，数据区损坏会在解码期
    /// 以数据错误暴露；需要强校验语义（损坏即报 blake3 错）用 [`Self::read`]。
    pub fn bytes(&self, name: &str) -> AvResult<&[u8]> {
        let e = self
            .entry(name)
            .ok_or_else(|| AvError::data(format!("容器中无条目: {name}")))?;
        self.entry_bytes(e)
    }

    /// 条目字节（零拷贝借用，无校验；调用方通常经 [`Self::bytes`] 按名获取）。
    pub fn entry_bytes(&self, e: &Entry) -> AvResult<&[u8]> {
        let start = (self.data_start + e.offset) as usize;
        let end = start + e.len as usize;
        if end > self.map.len() {
            return Err(AvError::data(format!("条目 {} 数据越界", e.name)));
        }
        Ok(&self.map[start..end])
    }

    /// 按名字读取文件内容，并校验该条目 blake3（拷贝返回；训练加载器走
    /// [`Self::bytes`] 零拷贝路径）。
    pub fn read(&self, name: &str) -> AvResult<Vec<u8>> {
        let e = self
            .entry(name)
            .ok_or_else(|| AvError::data(format!("容器中无条目: {name}")))?;
        let bytes = self.entry_bytes(e)?;
        let mut h = Hasher::new();
        h.update(bytes);
        if h.finalize().as_bytes() != &e.hash {
            return Err(AvError::data(format!("条目 {name} blake3 校验失败")));
        }
        Ok(bytes.to_vec())
    }
}

/// 便捷函数：读整个容器为 名 → 字节 的映射（小容器/测试用）。
pub fn read_all(path: &Path) -> AvResult<BTreeMap<String, Vec<u8>>> {
    let r = AvPackReader::open(path)?;
    let mut m = BTreeMap::new();
    for e in r.entries() {
        m.insert(e.name.clone(), r.read(&e.name)?);
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_source(dir: &Path) -> AvResult<()> {
        std::fs::create_dir_all(dir.join("images/train"))?;
        std::fs::create_dir_all(dir.join("labels/train"))?;
        std::fs::write(dir.join("images/train/a.jpg"), b"\xFF\xD8fake-jpeg-a")?;
        std::fs::write(dir.join("images/train/b.png"), b"\x89PNG-fake-b")?;
        std::fs::write(dir.join("labels/train/a.txt"), "0 0.1 0.2 0.3 0.4\n")?;
        std::fs::write(dir.join("labels/train/b.txt"), "5 0.5 0.5 0.2 0.2\n")?;
        Ok(())
    }

    #[test]
    fn pack_roundtrip_preserves_content_and_order() {
        let dir = std::env::temp_dir().join("avpack-test-src");
        let _ = std::fs::remove_dir_all(&dir);
        make_source(&dir).unwrap();
        let out = dir.parent().unwrap().join("test.avpack");
        let (count, bytes) = pack_dir(&dir, &out).unwrap();
        assert_eq!(count, 4);
        assert!(bytes > 0);

        let r = AvPackReader::open(&out).unwrap();
        assert_eq!(r.entries().len(), 4);
        let names: Vec<&str> = r.entries().iter().map(|e| e.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "images/train/a.jpg",
                "images/train/b.png",
                "labels/train/a.txt",
                "labels/train/b.txt"
            ]
        );
        assert_eq!(
            r.read("images/train/a.jpg").unwrap(),
            b"\xFF\xD8fake-jpeg-a"
        );
        assert_eq!(
            r.read("labels/train/b.txt").unwrap(),
            b"5 0.5 0.5 0.2 0.2\n"
        );
        assert!(r.read("nope.txt").is_err());
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn corrupted_entry_hash_is_detected() {
        let dir = std::env::temp_dir().join("avpack-test-src2");
        let _ = std::fs::remove_dir_all(&dir);
        make_source(&dir).unwrap();
        let out = dir.parent().unwrap().join("corrupt.avpack");
        pack_dir(&dir, &out).unwrap();

        // 篡改数据区一个字节（数据区起点 = HEADER_LEN）
        let mut raw = std::fs::read(&out).unwrap();
        let i = HEADER_LEN as usize + 5;
        raw[i] ^= 0xFF;
        std::fs::write(&out, &raw).unwrap();

        let r = AvPackReader::open(&out).unwrap();
        let mut corrupted = Vec::new();
        for e in r.entries() {
            if r.read(&e.name).is_err() {
                corrupted.push(e.name.clone());
            }
        }
        assert_eq!(corrupted.len(), 1, "恰好一个条目应校验失败: {corrupted:?}");
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn truncated_index_is_rejected_not_panic() {
        // 模拟"索引哈希一致但 count 虚高"的截断/恶意容器：重写索引区 count
        // 为 9999 并重算 blake3 回填头——open 必须报数据错误而非切片 panic
        // （release inference profile panic=abort，panic 即进程崩溃）。
        let dir = std::env::temp_dir().join("avpack-test-src3");
        let _ = std::fs::remove_dir_all(&dir);
        make_source(&dir).unwrap();
        let out = dir.parent().unwrap().join("trunc.avpack");
        pack_dir(&dir, &out).unwrap();
        let raw = std::fs::read(&out).unwrap();

        let index_offset = u64::from_le_bytes(raw[12..20].try_into().unwrap()) as usize;
        let index_len = u64::from_le_bytes(raw[20..28].try_into().unwrap()) as usize;
        let mut index = raw[index_offset..index_offset + index_len].to_vec();
        index[0..4].copy_from_slice(&9999u32.to_le_bytes());
        let mut ih = Hasher::new();
        ih.update(&index);
        let hash: [u8; 32] = ih.finalize().into();

        let mut forged = raw[..index_offset].to_vec();
        forged.extend_from_slice(&index);
        forged[28..60].copy_from_slice(&hash);
        std::fs::write(&out, &forged).unwrap();

        let res = AvPackReader::open(&out);
        assert!(res.is_err(), "截断索引必须报错而非 panic");
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn mmap_zero_copy_path_matches_read() {
        // mmap 路径（bytes/entry_bytes/entry）与校验拷贝路径（read）逐字节一致
        let dir = std::env::temp_dir().join("avpack-test-src4");
        let _ = std::fs::remove_dir_all(&dir);
        make_source(&dir).unwrap();
        let out = dir.parent().unwrap().join("mmap.avpack");
        pack_dir(&dir, &out).unwrap();

        let r = AvPackReader::open(&out).unwrap();
        for e in r.entries() {
            let borrowed = r.bytes(&e.name).unwrap();
            let verified = r.read(&e.name).unwrap();
            assert_eq!(borrowed, &verified[..], "条目 {} 两路径应一致", e.name);
            assert_eq!(r.entry(&e.name).unwrap().name, e.name);
        }
        assert!(r.bytes("nope.bin").is_err());
        assert!(r.entry("nope.bin").is_none());
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn header_truncated_file_is_rejected() {
        let dir = std::env::temp_dir().join("avpack-test-src5");
        let _ = std::fs::remove_dir_all(&dir);
        make_source(&dir).unwrap();
        let out = dir.parent().unwrap().join("short.avpack");
        pack_dir(&dir, &out).unwrap();
        // 截到不足一个文件头：open 必须报错（mmap 前的长度检查，非空文件才可映射）
        let raw = std::fs::read(&out).unwrap();
        std::fs::write(&out, &raw[..16]).unwrap();
        assert!(AvPackReader::open(&out).is_err());
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_file(&out);
    }
}
