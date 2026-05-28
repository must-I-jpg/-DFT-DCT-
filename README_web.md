# 图像水印网页端

这个网页端基于项目里的 DFT 频域水印思路：用户上传载体图片和水印图片，后端自动二值化水印并生成含水印图片。

## 启动

```bash
python app.py
```

浏览器打开：

```text
http://127.0.0.1:5000
```

## 依赖

如果缺依赖，安装：

```bash
pip install -r requirements-web.txt
```

## 文件

- `app.py`：Flask 后端和 DFT 水印嵌入逻辑
- `templates/index.html`：上传和结果页面
- `static/styles.css`：页面样式
- `web_outputs/`：生成后的含水印图片
