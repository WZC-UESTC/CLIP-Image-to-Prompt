# BLIP + Language Model Comparison Report

## Summary Table

| Image | BLIP Input | T5 | BART | GPT2 | CLIP-Interrogator |
|-------|------------|----|----|------|-------------------|
| sunset.jpg | a sunset over the ocean | a beautiful sunset casting golden light across the... | The sun sets over the calm ocean, painting the sky... | a stunning sunset over the ocean with vibrant colo... | a breathtaking sunset over the ocean, golden hour,... |
| cat.jpg | a cat sitting on a windowsill | a domestic cat resting on a window ledge, looking ... | A cat is sitting on a windowsill, gazing through t... | a cute cat sitting on a windowsill, looking outsid... | a cute domestic cat sitting on a windowsill, natur... |
| dog.jpg | a dog running on grass | a canine running across a green field with energy | A dog is running playfully on the green grass fiel... | a happy dog running on green grass in a sunny park | a happy golden retriever running on green grass, s... |
| building.jpg | a tall building in the city | a skyscraper towering over an urban landscape | A tall building stands prominently in the city sky... | a modern skyscraper in the middle of a busy city | a modern skyscraper in downtown city, urban archit... |
| food.jpg | a plate of food on a table | a meal arranged on a dining table with various dis... | A plate of delicious food is placed on the wooden ... | a plate of fresh food on a wooden table, ready to ... | a delicious plate of gourmet food, wooden table se... |

## Detailed Results

### sunset.jpg

![sunset.jpg](./test_images\sunset.jpg)

- **BLIP Input:** a sunset over the ocean
- **T5 Output:** a beautiful sunset casting golden light across the ocean horizon
- **BART Output:** The sun sets over the calm ocean, painting the sky in warm orange and pink hues.
- **GPT2 Output:** a stunning sunset over the ocean with vibrant colors reflecting on the water
- **CLIP-Interrogator:** a breathtaking sunset over the ocean, golden hour, dramatic sky, warm colors, peaceful atmosphere, high resolution, beautiful landscape photography

---

### cat.jpg

![cat.jpg](./test_images\cat.jpg)

- **BLIP Input:** a cat sitting on a windowsill
- **T5 Output:** a domestic cat resting on a window ledge, looking outside
- **BART Output:** A cat is sitting on a windowsill, gazing through the glass at the outside world.
- **GPT2 Output:** a cute cat sitting on a windowsill, looking outside at the birds
- **CLIP-Interrogator:** a cute domestic cat sitting on a windowsill, natural lighting, looking outside, curious expression, detailed fur texture, indoor setting, cozy atmosphere

---

### dog.jpg

![dog.jpg](./test_images\dog.jpg)

- **BLIP Input:** a dog running on grass
- **T5 Output:** a canine running across a green field with energy
- **BART Output:** A dog is running playfully on the green grass field.
- **GPT2 Output:** a happy dog running on green grass in a sunny park
- **CLIP-Interrogator:** a happy golden retriever running on green grass, sunny day, motion blur, energetic pose, fluffy coat, outdoor park setting, vibrant colors

---

### building.jpg

![building.jpg](./test_images\building.jpg)

- **BLIP Input:** a tall building in the city
- **T5 Output:** a skyscraper towering over an urban landscape
- **BART Output:** A tall building stands prominently in the city skyline.
- **GPT2 Output:** a modern skyscraper in the middle of a busy city
- **CLIP-Interrogator:** a modern skyscraper in downtown city, urban architecture, glass facade, cloudy sky, cityscape background, professional photography, sharp details

---

### food.jpg

![food.jpg](./test_images\food.jpg)

- **BLIP Input:** a plate of food on a table
- **T5 Output:** a meal arranged on a dining table with various dishes
- **BART Output:** A plate of delicious food is placed on the wooden table.
- **GPT2 Output:** a plate of fresh food on a wooden table, ready to eat
- **CLIP-Interrogator:** a delicious plate of gourmet food, wooden table setting, appetizing presentation, fresh ingredients, natural lighting, food photography, vibrant colors

---

