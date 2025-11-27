In this project, we explore a technique known as [Procedural Stochastic Tiling](https://unity.com/blog/engine-platform/procedural-stochastic-texturing-in-unity), and break out the most basic pieces so that we can build them back and learn how they work.

<img width="320" height="320" alt="Unity_rtuomLUyxN" src="https://github.com/user-attachments/assets/b8d57fb2-3b2a-44fa-972e-e9c468167588" />
<img width="320" height="320" alt="Unity_rtuomLUyxN" src="https://github.com/user-attachments/assets/92d1e353-9e3c-43a2-8722-636372438563" />

Procedural Stochastic Tiling is a technique that allows us to reduce repetition in a Stochastic Texture by breaking it up into smaller segments and layering multiple different segments with differing rotations and mirrors. This project includes a simplified ShaderGraph node

<img width="480" height="321" alt="Unity_rtuomLUyxN" src="https://github.com/user-attachments/assets/9a3e5624-7292-4b00-aa28-5b94c3a2e8f6" />

as well as a mode advanced version with Normal mapping 

<img width="344" height="296" alt="Unity_2trZkNZBVJ" src="https://github.com/user-attachments/assets/4194d0ae-a5fb-4601-8b10-5495ed2a06a3" />

Stochastic Tiling unfortunately produces an edge artifact which can be seen in some cases

<img width="370" height="415" alt="Unity_03Gt9OiZEy" src="https://github.com/user-attachments/assets/1bff3137-da97-4267-bf21-cba532328d4c" />

The project proposes a "jittering" technique which shifts the edge point by simple 2d noise

![Unity_91ace5ukqO](https://github.com/user-attachments/assets/eb95e40c-ba4e-426d-a6a1-8c5395e8aced)>

This project also includes an even more advanced version which takes a Property map packed as (Height, AO, Smoothness) using a software such as [Materialize](https://github.com/BoundingBoxSoftware/Materialize)

<img width="831" height="389" alt="Unity_ELYQ7NCDQn" src="https://github.com/user-attachments/assets/b939ef69-d6fd-4d4e-bea8-a34d1921f4e4" />

The height property is used to ensure our tallest terrain breaks through, which gives better peaks for things like Grass and Rocks

![Unity_uugkfIFDFA](https://github.com/user-attachments/assets/37a9167e-ed7b-4686-abb6-c1e5342eb7b6)

The algorithm results in terrain which looks great from close and from afar

![Unity_BZpbnnDLd5](https://github.com/user-attachments/assets/02248bd9-4702-464c-b365-8ea2458de201)

The included Materials have a range of settings to control the effects

<img width="237" height="560" alt="Unity_zwOwIm8sMi" src="https://github.com/user-attachments/assets/2fc47cda-7de0-4cab-ae7a-4fa56b8c8133" />

Future versions of this project will tackle topics such as Triplanar mapping as well as SplatMap support for Unity Terrains with height-aware blending
