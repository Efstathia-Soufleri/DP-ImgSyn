from torchvision.datasets import CelebA
import torch

class CelebAGender(CelebA):
    def __init__(
            self, 
            root: str, 
            split: str = "train", 
            transform = None, 
            target_transform = None, 
            download: bool = False) -> None:
        super().__init__(
            root, 
            split, 
            target_type='attr', 
            transform=transform, 
            target_transform=target_transform,
            download=download)
        
        attr_names = [
            "5_o_Clock_Shadow",
            "Arched_Eyebrows",
            "Attractive",
            "Bags_Under_Eyes",
            "Bald",
            "Bangs",
            "Big_Lips",
            "Big_Nose",
            "Black_Hair",
            "Blond_Hair",
            "Blurry",
            "Brown_Hair",
            "Bushy_Eyebrows",
            "Chubby",
            "Double_Chin",
            "Eyeglasses",
            "Goatee",
            "Gray_Hair",
            "Heavy_Makeup",
            "High_Cheekbones",
            "Male",
            "Mouth_Slightly_Open",
            "Mustache",
            "Narrow_Eyes",
            "No_Beard",
            "Oval_Face",
            "Pale_Skin",
            "Pointy_Nose",
            "Receding_Hairline",
            "Rosy_Cheeks",
            "Sideburns",
            "Smiling",
            "Straight_Hair",
            "Wavy_Hair",
            "Wearing_Earrings",
            "Wearing_Hat",
            "Wearing_Lipstick",
            "Wearing_Necklace",
            "Wearing_Necktie",
            "Young"
        ]
        self.hair_idxs = [
            attr_names.index("Male") ]
    
    def __getitem__(self, index: int):
        data, attr = super().__getitem__(index)
        attr = attr[self.hair_idxs]
        attr2 = torch.clamp(attr, 0, 1).item()  # from -1, 1 => 0 , 1
        return data, attr2