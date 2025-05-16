import os
from skimage import io
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class LithosDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        polar_vit: bool = True,
        root_dataset: str = None,
        preprocess =  None, 
        transform=None,
        use_xpl = False,
        num_classes = 2
    ):
        """
        Constructor for the LithosDataset class.

        Args:
            df (pd.DataFrame): the pandas dataframe object
            polar_vit (bool): whether to use our polar vit model that uses the two polarizations (default is True)
            root_dataset (str): the root path to the dataset
            preprocess (torchvision.transforms): the preprocessing transformations
            transform (torchvision.transforms): the transformations to apply to the images
            use_xpl (bool): whether to use xpl polarizations or not
            num_classes (int): the number of classes. Right now LithosDataset supports binary (2) and multiclass (25) classes
        """
        if num_classes not in [2, 25]:
            raise ValueError(f"num_classes must be 2 or 25, but got {num_classes}")

        self.df = df
        self.polar_vit = polar_vit
        self.root_dataset = root_dataset
        self.preprocess = preprocess
        self.transform = transform
        self.toTensor = transforms.ToTensor()
        self.use_xpl = use_xpl
        self.num_classes = num_classes
        self.define_mapping()
        self.__checkdf__()

    def path_exists(self, path):
        return os.path.exists(os.path.join(self.root_dataset, path))

    def __checkdf__(self):
        self.df =  self.df[self.df['Sec_patch'].apply(self.path_exists)]

    def define_mapping(self):
        remap_dict = {4: 24, 21: 24, 25: 24, 26: 24}
        unique_classes = sorted(set(range(self.num_classes + len(remap_dict.keys()))) - set(remap_dict.keys()))
        mapping = {old_class: new_idx for new_idx, old_class in enumerate(unique_classes)}
        mapping.update(remap_dict)
        self.mapping = mapping

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path_ppl = row["Sec_patch"]
        directory = os.path.dirname(image_path_ppl)
        image_path_ppl = os.path.join(self.root_dataset, image_path_ppl)
        image_path_xpl0 = image_path_ppl.replace("ppl-0", "xpl-0")
        annotation_path = os.path.join(self.root_dataset, directory, "center_component.csv")
        annotation = pd.read_csv(annotation_path)
        binary_patch_class = annotation["Binary_label"][0]
        multiclass_patch_class = annotation["Multiclass_label"][0]
        
        if self.num_classes == 2:
            target = binary_patch_class
        else:
            target = self.mapping[multiclass_patch_class]

        # Use ppl and xpl images and polar vit
        if self.polar_vit:            
            if not os.path.exists(image_path_ppl) or not os.path.exists(image_path_xpl0):
                raise FileNotFoundError(f"Couldn't find image {image_path_ppl} or {image_path_xpl0}")

            image_ppl0 = io.imread(image_path_ppl)
            image_xpl0 = io.imread(image_path_xpl0)

            if self.transform is not None:
                image_ppl0 = self.transform(image_ppl0)
                image_xpl0 = self.transform(image_xpl0)
            else: # Used for testing
                image_ppl0 = self.toTensor(image_ppl0)
                image_xpl0 = self.toTensor(image_xpl0)

            if self.preprocess is not None:
                image_ppl0 = self.preprocess(image_ppl0)
                image_xpl0 = self.preprocess(image_xpl0)

            return {'data1': image_ppl0, 'data2': image_xpl0, 'target': target}
        
        # Use only one polarization and single modality models
        else: 
            # To use xpl polarization
            if self.use_xpl:
                image_path = image_path_xpl0
            else:
                image_path = image_path_ppl

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Couldn't find image {image_path}")
            
            image = io.imread(image_path)

            if self.transform is not None:
                image = self.transform(image)
            else: # Used for testing
                image = self.toTensor(image)

            if self.preprocess is not None:
                image = self.preprocess(image)
        
        return {'data': image, 'target': target}  