mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
normalizer = torchvision.transforms.Normalize(mean, std, inplace=False)
resizer = torchvision.transforms.Resize((224, 224))

inputs = normalizer(resizer(image))

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', cache_dir='./model/')

outputs = model(inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])

