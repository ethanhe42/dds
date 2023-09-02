# DDS: Delta Denoising Score PyTorch implementation
We introduce Delta Denoising Score (DDS), a novel scoring function for text-based image editing that guides minimal modifications of an input image towards the content described in a target prompt. DDS leverages the rich generative prior of text-to-image diffusion models and can be used as a loss term in an optimization problem to steer an image towards a desired direction dictated by a text. DDS utilizes the Score Distillation Sampling (SDS) mechanism for the purpose of image editing. We show that using only SDS often produces non-detailed and blurry outputs due to noisy gradients. To address this issue, DDS uses a prompt that matches the input image to identify and remove undesired erroneous directions of SDS. Our key premise is that SDS should be zero when calculated on pairs of matched prompts and images, meaning that if the score is non-zero, its gradients can be attributed to the erroneous component of SDS. Our analysis demonstrates the competence of DDS for text based image-to-image translation. We further show that DDS can be used to train an effective zero-shot image translation model. Experimental results indicate that DDS outperforms existing methods in terms of stability and quality, highlighting its potential for real-world applications in text-based image editing.
![image](https://github.com/yihui-he/dds/assets/10027339/69293e78-3a89-49b3-83fe-976e394dff7f)


| a flamingo rollerskating | a bronze sculpture of a flamingo rollerskating | 
|-----------------------|------------------------------------------------|
| ![image](https://github.com/yihui-he/dds/assets/10027339/b2bc6066-24aa-474f-88ff-b11e878a9703) | ![image](https://github.com/yihui-he/dds/assets/10027339/7f69c461-13d1-4694-b4eb-597a0677a0b1) | 


| a flamingo rollerskating | a stork rollerskating | 
|-----------------------|------------------------------------------------|
| ![image](https://github.com/yihui-he/dds/assets/10027339/b2bc6066-24aa-474f-88ff-b11e878a9703) | ![image](https://github.com/yihui-he/dds/assets/10027339/85330852-9223-4af5-82c7-5a4303b357e3) |

| a flamingo rollerskating | a giraffe rollerskating | 
|-----------------------|------------------------------------------------|
| ![image](https://github.com/yihui-he/dds/assets/10027339/b2bc6066-24aa-474f-88ff-b11e878a9703) | ![image](https://github.com/yihui-he/dds/assets/10027339/3256763e-0fa1-4434-8b42-f03ebf2f361c) |

