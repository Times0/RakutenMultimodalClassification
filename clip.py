from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F


def load_clip_model():
    """
    Load the CLIP model and processor
    """
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


def load_image(image_path):
    """
    Load image from path or URL
    """
    try:
        if image_path.startswith("http"):
            image = Image.open(requests.get(image_path, stream=True).raw)
        else:
            image = Image.open(image_path)
        return image
    except Exception as e:
        raise Exception(f"Error loading image: {str(e)}")


def generate_candidate_captions(base_concepts):
    """
    Generate candidate captions based on common concepts
    """
    templates = [
        "a photo of {}",
        "an image of {}",
        "a picture showing {}",
        "{} in the scene",
        "this is {}",
    ]

    candidates = []
    for concept in base_concepts:
        for template in templates:
            candidates.append(template.format(concept))
    return candidates


def get_best_caption(image_path, candidate_captions, model, processor):
    """
    Use CLIP to find the best matching caption for an image
    """
    # Load and preprocess image
    image = load_image(image_path)
    inputs = processor(
        text=candidate_captions, images=image, return_tensors="pt", padding=True
    )

    # Get features
    outputs = model(**inputs)
    image_features = outputs.image_embeds
    text_features = outputs.text_embeds

    # Calculate similarity scores
    similarity = F.cosine_similarity(image_features, text_features)

    # Get best matching caption
    best_idx = similarity.argmax().item()
    return candidate_captions[best_idx], similarity[best_idx].item()


def caption_image(image_path, concepts=None):
    """
    Main function to generate caption for an image
    """
    # Default concepts if none provided
    if concepts is None:
        concepts = [
            "landscape",
            "person",
            "animal",
            "building",
            "vehicle",
            "food",
            "indoor scene",
            "outdoor scene",
            "artwork",
            "natural scenery",
            "urban environment",
            "group of people",
            "plane",
            
        ]

    try:
        # Load model and processor
        model, processor = load_clip_model()

        # Generate candidate captions
        candidates = generate_candidate_captions(concepts)
        print(candidates)
        # Get best caption
        caption, confidence = get_best_caption(image_path, candidates, model, processor)

        return {"caption": caption, "confidence": confidence, "status": "success"}

    except Exception as e:
        return {"caption": None, "confidence": None, "status": "error", "error": str(e)}


# Example usage
if __name__ == "__main__":
    # Example image URL
    image_url = "https://cdn.openai.com/clip/assets/zero-shot/zs_32.png"

    # Optional: Custom concepts

    # Generate caption
    result = caption_image(image_url)

    if result["status"] == "success":
        print(f"Caption: {result['caption']}")
        print(f"Confidence: {result['confidence']:.2f}")
    else:
        print(f"Error: {result['error']}")
