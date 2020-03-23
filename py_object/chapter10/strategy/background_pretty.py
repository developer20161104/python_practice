# 执行三种缩放策略
from PIL import Image


# 不断重复复制图片，直到铺满
class TiledStrategy:
    def make_background(self, img_file, desktop_size):
        in_img = Image.open(img_file)

        out_img = Image.new('RGB', desktop_size)
        num_titles = [
            o // i + 1 for 0, i in
            zip(out_img.size, in_img.size)
        ]

        for x in range(num_titles[0]):
            for y in range(num_titles[1]):
                out_img.paste(
                    in_img,
                    (
                        in_img.size[0] * x,
                        in_img.size[1] * y,
                        in_img.size[0] * (x + 1),
                        in_img.size[1] * (y + 1),
                    )
                )

        return out_img


# 图片居中
class CenteredStrategy:
    def make_background(self, img_file, desktop_size):
        in_img = Image.open(img_file)
        out_img = Image.new('RGB', desktop_size)

        left = (out_img.size[0] - in_img.size[0]) // 2
        top = (out_img.size[1] - in_img.size[1]) // 2

        out_img.paste(
            in_img,
            (
                left,
                top,
                left + in_img.size[0],
                top + in_img.size[1],
            )
        )

        return out_img


class ScaledStrategy:
    def make_background(self, img_file, desktop_size):
        in_img = Image.open(img_file)
        out_img = in_img.resize(desktop_size)
        return out_img
