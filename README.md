# Facial Emotion Recognition

基于 2D 摄像头的网页版人脸面部情绪识别应用，覆盖：焦虑（Anxiety）、困惑（Confusion）、疲劳（Fatigue）、专注（Concentration）、兴奋（Excitement）、厌倦（Boredom）、压力（Stress）。

## 技术方案

- **面部感知**：使用 MediaPipe Face Landmarker 在浏览器侧实时提取 478 点人脸网格、52 维表情 blendshape 与头部姿态矩阵。
- **状态推理**：融合眉眼、嘴部、眨眼、张口、微笑、头部偏转等 2D/几何特征，输出七类状态概率。
- **一致性**：使用指数滑动平均与主状态滞回机制，减少每帧预测抖动。
- **实时性**：模型在浏览器中运行，可使用 GPU delegate；摄像头建议 720p/30fps。
- **鲁棒性**：人脸置信度、追踪置信度、脸部尺度质量评分和丢帧保留共同降低遮挡、偏头、弱光下误判。

> 说明：本项目是工程化实时情绪状态估计工具，不应作为医疗诊断、心理评估或用工/教育高风险自动决策的唯一依据。若需要“高准确率”生产部署，请用目标场景采集数据进行标注、校准和验证。

## 本地运行

```bash
npm install
npm run start
```

浏览器打开 `http://localhost:5173`，点击“启动摄像头”并授权摄像头权限。

## 构建

```bash
npm run build
```
