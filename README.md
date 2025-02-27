# Neural Style Transfer

This project implements Neural Style Transfer using PyTorch. NST applies the artistic style of one image to another while preserving its content.

This is Implementation of Paper Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization .

## Requirements

Ensure you have the following dependencies installed:

```sh
pip install torch torchvision numpy matplotlib Pillow
```

## Usage

Train the model using the provided Jupyter Notebook:

```sh
neural_style_transfer.ipynb
```

Run the model using:

```sh
python test.py
```

Run the website using Streamlit:

```sh
streamlit run app.py
```

## Results

The output image will be a combination of the content image and the artistic style of the style image.

![Neural Style Transfer Example](example.png)


