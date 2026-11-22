# 视频解码测试素材

`local-preview.webm` 是 160 × 96、1.2 秒的程序生成色块测试图，仅用于验证浏览器解码、抽样、取消和数据来源隔离，不含用户采集或研究结果。

生成命令：

```bash
ffmpeg -f lavfi -i testsrc2=size=160x96:rate=10 -t 1.2 -c:v libvpx -b:v 60k -an local-preview.webm
```
