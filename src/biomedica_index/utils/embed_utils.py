try:
    import torchvision.transforms as tvt
    IMAGE_DISABLED = False
except ImportError:
    IMAGE_DISABLED = True

try:
    from transformers import AutoTokenizer, AutoModel
    TEXT_DISABLED = False
except ImportError:
    TEXT_DISABLED = True

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    TEXT_DISABLED = True
    IMAGE_DISABLED = True

class SentenceEmbedder:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', hf_token=None, device=None):
        if TEXT_DISABLED:
            print("WARNING: Cannot query by text embeddings because necessary packages are not installed")
        else:
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
            self.model = AutoModel.from_pretrained(model_name, token=hf_token).to(self.device)
        
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode(self, sentences):
        with torch.inference_mode():
            encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            return F.normalize(sentence_embeddings, p=2, dim=1).squeeze().cpu().numpy()

    def preprocess(self, sentences):
        return sentences

class ImageEmbedder:
    BATCH_SIZE = 128

    def __init__(self, dino_version='dinov2_vitl14_reg', device=None):
        if IMAGE_DISABLED:
            print("WARNING: Cannot query by image embeddings because necessary packages are not installed")
        else:
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = torch.hub.load('facebookresearch/dinov2', dino_version)
            self.model.eval().to(self.device)
            self.resize = tvt.Compose([ # assumes input is PIL image
                tvt.Resize((224,224)),  # Resize 
                tvt.ToTensor()           # Convert PIL Image to PyTorch Tensor
            ])
    
    def encode(self, images):
        with torch.inference_mode():
            if isinstance(images, list):
                images = torch.stack(images)
            img = images.to(self.device)
            if len(img.shape) < 4:
                img = img.unsqueeze(0)
            features = self.model(img).squeeze().cpu().numpy()
            return features

    def preprocess(self, images):
        return [self.resize(img) for img in images]
