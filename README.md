# CharFormer

This project refers to the code and experiments of the ACM MM 2022 Accepted paper "[CharFormer: A Glyph Fusion based Attentive Framework for High-precision Character Image Denoising](https://dl.acm.org/doi/abs/10.1145/3503161.3548208)". 

### Abstract:
Degraded images commonly exist in the general sources of character images, leading to unsatisfactory character recognition results. Existing methods have dedicated efforts to restoring degraded character images. However, the denoising results obtained by these methods do not appear to improve character recognition performance. This is mainly because current methods only focus on pixel-level information and ignore critical features of a character, such as its glyph, resulting in character-glyph damage during the denoising process. In this paper, we introduce a novel generic framework based on glyph fusion and attention mechanisms, i.e., CharFormer, for precisely recovering character images without changing their inherent glyphs. Unlike existing frameworks, CharFormer introduces a parallel target task for capturing additional information and injecting it into the image-denoising backbone, which will maintain the consistency of character glyphs during character image denoising. Moreover, we utilize attention-based networks for global-local feature interaction, which will help to deal with blind denoising and enhance denoising performance. 

### Project Introduction

##### models: the proposed glyph fusion based model;
##### util: some supportive functions to implement model training and testing;
##### datasets: the function to load data to model;
##### predict: character image denoising;
##### train: training of our denoising model.
