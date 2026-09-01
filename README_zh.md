# Z1KV

嵌入式 MVCC key-value 存储引擎(纯库 crate,无 bin 入口)。

- **版本模型**:每个版本 `(cf, key, txn_id) -> value | tombstone`
- **可见性**:严格快照隔离(`commit_ts_map` 缺失即不可见,规则 D12)+ SSI 冲突检测
- **存储**:三层增量栈——L1 内存(ping-pong + RCU 读)→ L2 磁盘 patch → L3 冻结(GC 归并)
- **持久化**:自实现 WAL(WAL-first,`append_durable` 是唯一持久化边界)+ 崩溃安全 checkpoint
- **不变量**:D4 / D5 / D7 / D8 / D12(定义见 `src/lib.rs`)

## 快速上手

在 `Cargo.toml` 中添加:

```toml
[dependencies]
z1kv = "0.1"
```

```rust
use z1kv::Z1Kv;

// 打开(或创建)引擎;进程级锁保证同一数据目录不被双开。
let db = Z1Kv::open("my-data-dir")?;

// 事务写。
let txn = db.begin_txn()?;
db.put(0, b"greeting", b"hello", txn)?;        // cf, key, value, txn
db.commit(txn)?;

// 读:当前快照。
assert_eq!(db.get(0, b"greeting")?, Some(b"hello".to_vec()));

// 事务内读:begin 时固定快照,可重复读,参与 SSI 冲突检测。
let t1 = db.begin_txn()?;
assert_eq!(db.get_for_txn(0, b"greeting", t1)?, Some(b"hello".to_vec()));
db.commit(t1)?;

// 删除(墓碑)。
let t2 = db.begin_txn()?;
db.delete(0, b"greeting", t2)?;
db.commit(t2)?;
assert_eq!(db.get(0, b"greeting")?, None);
```

## 范围扫描

```rust
// [start, end) 半开区间;end 为 None 表示无上界。
let rows = db.scan(0, b"a", Some(b"z"))?;   // Vec<(Vec<u8>, Vec<u8>)>,按 key 升序
```

## 维护

```rust
db.flush_now()?;        // 无条件 L1 → L2(db.flush() 是阈值触发的,小数据为 no-op)
db.checkpoint()?;       // flush → 写 checkpoint → 截断 WAL(崩溃安全顺序)
let (cfs, reclaimed) = db.compact(u64::MAX)?; // L2 → L3 归并 + GC(水位 = 最老活跃/固定快照)
```

## 重要契约

| 契约 | 说明 |
|---|---|
| 持久化边界 | `commit()` 返回 Ok 即 WAL 已 fsync;crash 后已提交写入必可恢复,未提交写入被丢弃 |
| 固定快照 | `begin_txn` 的事务快照参与 GC 水位,跨 compaction 稳定 |
| 裸快照 | `db.snapshot()` 是时间旅行读,其可见性**不被** GC 保证(见 `snapshot` doc) |
| 引擎锁 | 同一数据目录双开会显式报错(锁文件 `ENGINE.lock`,随实例 Drop 释放) |
| 记录上限 | 单条 WAL 记录 ≤ 64MB;超大 value 在 `put` 边界报错,不会写出损坏文件 |
| strict_mode | 默认开启:恢复时遇到损坏记录中止打开(而非静默跳过) |

## 配置

```rust
use z1kv::config::{Z1Config, VisibilityConfig};
use z1kv::Z1Kv;

let cfg = Z1Config::default()
    .with_checkpoint_wal_size_threshold(64 * 1024 * 1024) // WAL 超限自动 checkpoint;0 = 禁用
    .with_l2_compaction_threshold(64)                     // L2 patch 数超限自动 compaction;0 = 禁用
    .with_strict_mode(true)                               // 降级错误升级为致命
    .with_visibility(VisibilityConfig::default());        // 历史淘汰(数量/TTL)

let db = Z1Kv::open_with_config("my-data-dir", cfg)?;
```

> `Z1Config` 与 `VisibilityConfig` 是 `#[non_exhaustive]`:请用
> `default()` + `with_*` builder 方法构造,不要用结构体字面量。

## 架构

```
            ┌───────────────────────────────┐
   写入     │  Z1Kv (facade)                │
            │  ┌─────────┐    ┌──────────┐  │
   txn ───► │  │ MVCC    │◄──►│ WAL      │  │   D4: WAL-first
            │  └────┬────┘    └────┬─────┘  │
            │       │  L1 MemStore │        │   ping-pong hot/cold + ArcSwap RCU 读
            │       ▼              │        │
            │  ┌─────────┐   ┌─────▼─────┐  │   D8: recent-flush 缓存桥接竞态窗口
            │  │ L2 disk │──►│ L3 frozen │  │   compaction + GC
            │  └─────────┘   └───────────┘  │
            └───────────────────────────────┘
                   崩溃安全 checkpoint → 截断 WAL
```

## 测试

```sh
cargo test          # 单元 + 集成 + README 验证 + doc 测试
cargo test --all-targets --release
```

测试套件包含 WAL 字节级崩溃矩阵(逐边界截断与位翻转注入)、checkpoint
envelope 翻转矩阵、GC 保守性与可见性双实现等价的属性测试(proptest)、
fuzz 契约冒烟测试,以及并发压测。

## 许可证

基于 [Apache License, Version 2.0](LICENSE) 授权。
