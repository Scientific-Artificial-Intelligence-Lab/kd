# Evolution Visualization 开发文档

## 架构设计

### 类结构
```
EvolutionVisBase (基类)
├── EvolutionAnimation (实时动画)
└── EvolutionSnapshot (序列帧生成)
```

### 布局设计
```
+-----------------+
|    Population   |
|    Evolution    |
+--------+--------+
| Best   | Stats  |
| Indiv. | Info   |
+--------+--------+
```

## 功能实现状态

### 核心功能完成度
- ✅ 基础可视化框架
  - Population distribution
  - Best individual display
  - Statistics panel
- ✅ 多格式输出支持
  - MP4 视频生成
  - GIF 动画生成
  - Frames 序列输出
- ✅ 并行处理优化
  - 动态 worker 调整
  - 智能 chunk 分配
- ✅ 自适应帧率
  - 基于时长的 fps 计算
  - 合理的帧率范围限制
- ⚠️ 实时动画模式
  - 基础动画功能完成
  - 缺少控制界面

### 性能优化状态
- ✅ 并行处理
  - 动态 worker 数量调整
  - 智能 chunk size 计算
  - 内存使用优化
- ✅ 文件管理
  - 临时文件自动清理
  - 元数据保留
  - 自定义路径支持
- ✅ 渲染优化
  - 临时帧低 DPI 渲染
  - 最终输出高质量保证

### 代码质量状态
- ✅ 结构设计
  - 清晰的类层次结构
  - 合理的职责分离
- ✅ 文档化
  - 完整的方法注释
  - 丰富的使用示例
- ⚠️ 待优化点
  - 部分重复代码
  - 错误处理可完善
  - 类型提示待补充

## 功能实现

### ✅ 已完成

#### 1. 基础可视化 (`EvolutionVisBase`)
- **Population Distribution**
  - Violin plot 展示分布
  - Scatter points 显示个体
  - 自动更新坐标轴
  
- **Best Individual Display**
  - 方程显示
  - Fitness 值
  - Complexity 指标
  
- **Statistics Panel**
  - Mean fitness
  - Standard deviation
  - Population diversity
  - Time tracking

#### 2. 序列帧生成 (`EvolutionSnapshot`)
- **多格式输出**
  ```python
  # MP4 视频输出
  plotter.plot_evolution(data, output_format="mp4", desired_duration=15)
  
  # GIF 动画输出
  plotter.plot_evolution(data, output_format="gif", desired_duration=10)
  
  # 图片序列输出
  plotter.plot_evolution(data, output_format="frames")
  ```

- **文件管理**
  - 自动创建临时目录
  - 自动清理临时文件
  - 支持自定义路径
  - 保留文件元数据

#### 3. 并行处理
- **动态优化**
  ```python
  # 自动计算最优配置
  max_workers = cpu_count * 2 if total_frames > 1000 else cpu_count
  chunk_size = min(total_frames // (max_workers * 2), 20)
  ```
- **内存管理**
  - 分块处理大数据集
  - 动态调整 worker 数量
  - 限制最大 chunk size

#### 4. 视频生成优化
- **帧率自适应**
  - 基于期望时长自动计算 fps
  - fps 限制在 1-30 范围内
  - 支持自定义时长

### 🚧 进行中

#### 1. 实时动画模式
- [ ] 播放控制
  - Play/Pause 按钮
  - 速度调节滑块
  - 帧跳转功能
- [ ] 实时更新优化
  - 减少重绘开销
  - 提高刷新效率

### 📊 测试覆盖

#### 已完成测试
```python
def test_snapshot_mp4(self):
    """Test snapshot mode with MP4 output."""
    result_path = self.plotter.plot_evolution(
        self.mock_data,
        mode="snapshot",
        output_format="mp4",
        desired_duration=5
    )
    # 验证视频时长
    with VideoFileClip(result_path) as clip:
        self.assertAlmostEqual(clip.duration, 5, delta=0.5)

def test_snapshot_frames(self):
    """Test frame sequence output."""
    result_path = self.plotter.plot_evolution(
        self.mock_data,
        mode="snapshot",
        output_format="frames"
    )
    frames = sorted(Path(result_path).glob("evolution_frame_*.png"))
    self.assertEqual(len(frames), len(self.mock_data))
```

#### 待补充测试
- [ ] 性能测试
  - 大数据集处理
  - 内存使用监控
  - 处理时间测试
- [ ] 边界条件测试
  - 空数据处理
  - 异常数据处理
  - 资源限制测试

## 使用示例

### 基本用法
```python
from kd.plot.interface.dlga import DLGAPlotter

# 创建 plotter
plotter = DLGAPlotter()

# 生成动画
plotter.plot_evolution(
    evolution_data,
    mode="animation",
    interval=100,  # 动画间隔
    figsize=(10, 8),  # 图像大小
    dpi=100  # 分辨率
)

# 生成视频
plotter.plot_evolution(
    evolution_data,
    mode="snapshot",
    output_format="mp4",
    output_path="evolution.mp4",
    desired_duration=15  # 期望时长(秒)
)
```

### 高级配置
```python
# 自定义临时目录和清理选项
plotter.plot_evolution(
    evolution_data,
    mode="snapshot",
    temp_dir=".custom_temp",
    cleanup_temp=True,
    output_format="mp4"
)

# 并行处理配置
plotter.plot_evolution(
    evolution_data,
    mode="snapshot",
    max_workers=4,
    chunk_size=10
)
```

## 开发计划

### 1. Performance Optimization
- [x] 实现动态 worker 数量调整
- [x] 优化 chunk size 计算
- [x] 添加内存监控机制

### 2. Animation Controls
- [ ] 添加播放控制界面
- [ ] 实现帧率调节功能
- [ ] 添加进度条显示

### 3. Testing
- [x] 基础功能测试
- [ ] 补充性能测试
- [ ] 完善边界条件测试

## 技术债务

### 1. 代码优化
- [ ] 提取公共组件
- [ ] 优化类结构
- [ ] 完善错误处理

### 2. 文档完善
- [x] 更新 API 文档
- [x] 添加更多使用示例
- [x] 完善注释说明

## 已知问题

### 1. 依赖库警告
- moviepy 库使用了旧版本 imageio API
- 不影响功能，等待 moviepy 更新
- 可能需要提交 PR 到 moviepy

### 2. 内存管理
- 大数据集处理时需要更细粒度的内存监控
- 考虑添加内存使用警告机制
- 可以实现更智能的内存回收

### 3. 错误处理
- 需要更详细的错误信息
- 可以添加错误恢复机制
- 异常情况的处理可以更完善

### 4. 实时动画
- 需要完善控制界面
- 考虑添加更多交互功能
- 性能优化空间较大 