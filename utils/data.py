import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImageReconstructionDataset(Dataset):
    """
    Dataset class for loading and preprocessing images for image reconstruction tasks.
    """

    def __init__(self, root_dir, image_size=(480, 640)):
        """
        Initialize the dataset.

        Args:
            root_dir (str): Path to the directory containing the images.
            image_size (tuple): Desired size of the images (height, width). Default is (512, 512).
        """
        #self.root_dir = root_dir
        if isinstance(root_dir, str):
            root_dir = [root_dir]
        self.root_dirs = root_dir
        
        self.image_size = image_size

        # Define transformations
        self.transform = transforms.Compose([
            transforms.Resize(image_size),  # Resize images to the specified size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
        ])
        
        self.image_files = []
        for directory in self.root_dirs:
            files_in_dir = [
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff'))
            ]
            self.image_files.extend(sorted(files_in_dir))

        # Ensure there are images in the directory
        if not self.image_files:
            raise RuntimeError(f"No images found in the directory: {root_dir}")

    def __len__(self):
        """
        Return the total number of images in the dataset.

        Returns:
            int: Number of images.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Load and preprocess an image from the dataset.

        Args:
            idx (int): Index of the image to load.

        Returns:
            tuple: A tuple containing the transformed image and itself (for reconstruction tasks).
        """
        # Get the image path
        img_path = self.image_files[idx]

        # Load the image and ensure it is in RGB format
        image = Image.open(img_path).convert("RGB")

        # Apply transformations
        image_transformed = self.transform(image)

        # For reconstruction tasks, return the same image as input and target
        return image_transformed, image_transformed


