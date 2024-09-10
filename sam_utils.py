import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import PIL

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def show_boxes_on_image(raw_image, boxes, save_on=None, title=""):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.title(f'Detection prompt: {title}') 
    if save_on is not None:
        plt.savefig(save_on)  
    else:
        plt.show()  

    plt.close() 

# def show_points_on_image(raw_image, input_points, input_labels=None, save_on=None):
#     plt.figure(figsize=(10,10))
#     plt.imshow(raw_image)
#     input_points = np.array(input_points)
#     if input_labels is None:
#       labels = np.ones_like(input_points[:, 0])
#     else:
#       labels = np.array(input_labels)
#     show_points(input_points, labels, plt.gca())
#     plt.axis('on')
#     plt.show()

def show_points_on_image(raw_image, input_points, input_labels=None, save_on=None, title="None"):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    
    if input_labels is None:
        labels = np.ones_like(input_points[:, 0])
    else:
        labels = np.array(input_labels)
    
    show_points(input_points, labels, plt.gca())
    plt.axis('on')
    plt.title(f'Detection prompt: {title}') 
    
    if save_on is not None:
        plt.savefig(save_on)  
    else:
        plt.show()  

    plt.close() 
    
    
def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()


# def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
#     plt.figure(figsize=(10,10))
#     plt.imshow(raw_image)
#     input_points = np.array(input_points)
#     if input_labels is None:
#       labels = np.ones_like(input_points[:, 0])
#     else:
#       labels = np.array(input_labels)
#     show_points(input_points, labels, plt.gca())
#     for box in boxes:
#       show_box(box, plt.gca())
#     plt.axis('on')
#     plt.show()


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=2)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_masks_on_image(raw_image, masks, scores, only_mask=False, segment_type=None, save_on=None):
    text = "" if segment_type is None else f"Using {segment_type}"
    if len(masks.shape) == 4:
      masks = masks.squeeze()
    if scores.shape[0] == 1:
      scores = scores.squeeze()

    nb_predictions = scores.shape[-1]
    fig, axes = plt.subplots(1, nb_predictions, figsize=(15, 15))
    for i, (mask, score) in enumerate(zip(masks, scores)):
      mask = mask.cpu().detach()
      if not only_mask:
        axes[i].imshow(np.array(raw_image))
      show_mask(mask, axes[i])
      axes[i].title.set_text(f"Mask {i+1}, Score: {score.item():.3f} "+text)
      axes[i].axis("off")
      
    if save_on is not None:
        plt.savefig(save_on)  
    else:
        plt.show()  
    
def resize_pil_image(img, ratio=0.5):
  image = img.copy()
  width, height = image.size

  downscaled_width = int(width * ratio)
  downscaled_height = int(height * ratio)

  image = image.resize((downscaled_width, downscaled_height))
  return image
  
  
def find_center(box):
    x_c = round(((box[2] - box[0])/2)+ box[0])
    y_c  = round(((box[3] - box[1])/2)+ box[1])
    return [x_c, y_c]
  
def load_image(path: str, resizing_ratio: float = 0.2) -> PIL.Image:
    raw_image = Image.open(path).convert("RGB")
    raw_image = ImageOps.exif_transpose(raw_image)
    raw_image = resize_pil_image(raw_image, ratio=resizing_ratio)
    return raw_image