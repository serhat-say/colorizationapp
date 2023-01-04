import argparse
import streamlit as st
import matplotlib.pyplot as plt
import time

from colorizers import *

st.title('Colorize App')



uploaded_img = st.file_uploader("please upload your jpeg file", type=["jpg"], accept_multiple_files=False)
time.sleep(3)

if uploaded_img is not None:
    img = load_img(uploaded_img)

colorizer_siggraph17 = siggraph17(pretrained=True).eval()

(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))


# colorizer outputs 256x256 ab map
# resize and concatenate to original L channel
img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))

out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

# Display images
st.subheader('Input')
st.image(img_bw)

st.subheader('Output')
st.image(out_img_siggraph17)