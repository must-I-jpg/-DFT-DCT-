# 图像水印网页端

这个网页端支持 DFT、DCT QIM 和 Haar DWT 水印嵌入：用户上传载体图片和水印图片，选择算法后，后端自动二值化水印并生成含水印图片。

## 开发模式启动

```bash
python3 app.py
```

程序会自动打开浏览器。也可以在终端输出中找到本机访问地址。

生成的图片保存在用户目录下的 `.frequency_watermark/web_outputs/` 中。

## 打包为双击运行的桌面程序

先安装运行依赖和打包工具：

```bash
python3 -m pip install -r requirements-build.txt
```

然后执行：

```bash
python3 build_desktop.py
```

PyInstaller 只能为当前操作系统打包，不能在 macOS 上直接生成 Windows 程序。

- 在 macOS 上运行后，双击 `dist/图像频域水印.app`。
- 在 Windows 上运行后，双击 `dist/图像频域水印.exe`。
- 在 Linux 上运行后，双击或执行 `dist/图像频域水印`。

程序启动后会自动打开浏览器页面。使用结束时点击页面左侧的“退出应用”。

## 文件

- `app.py`：Flask 后端和多种水印嵌入逻辑
- `build_desktop.py`：PyInstaller 桌面打包脚本
- `templates/index.html`：上传和结果页面
- `static/styles.css`：页面样式
- `requirements-web.txt`：运行依赖
- `requirements-build.txt`：运行依赖和打包工具
