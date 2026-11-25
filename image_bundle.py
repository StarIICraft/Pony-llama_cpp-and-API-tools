import torch


IMAGE_BUNDLE_TYPE = "PONY_IMAGE_BUNDLE"
MAX_BUNDLE_IMAGES = 10


class PonyImageInputBundle:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                f"image_{index}": ("IMAGE",)
                for index in range(1, MAX_BUNDLE_IMAGES + 1)
            },
        }

    RETURN_TYPES = (IMAGE_BUNDLE_TYPE,)
    RETURN_NAMES = ("图像包",)
    FUNCTION = "bundle"
    CATEGORY = "Pony Toolbox/Image"

    DESCRIPTION = (
        "把最多10路动态图像集中到一个稳定输出，"
        "供本地指令、API LLM、API图像和多图统一缩放节点使用。"
    )

    def bundle(self, **kwargs):
        images = []

        def visit(value, name):
            if value is None:
                return
            if isinstance(value, (list, tuple)):
                for item in value:
                    visit(item, name)
                return
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"{name} 必须是 ComfyUI IMAGE 张量，"
                    f"实际收到 {type(value).__name__}。"
                )
            if value.ndim not in (3, 4):
                raise ValueError(
                    f"{name} 的形状无效：{tuple(value.shape)}。"
                )
            if value.ndim == 4 and value.shape[0] == 0:
                return
            images.append(value)

        for index in range(1, MAX_BUNDLE_IMAGES + 1):
            name = f"image_{index}"
            visit(kwargs.get(name), name)

        return (images,)


NODE_CLASS_MAPPINGS = {
    "PonyImageInputBundle": PonyImageInputBundle,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PonyImageInputBundle": "动态图像输入-小马",
}
