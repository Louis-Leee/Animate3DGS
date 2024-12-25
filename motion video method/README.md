<b>Motion Video generation</b>
We want to generate motion videos given the initial image(s) and user prompt, so we can extract the trajectory from the videos. This is actually an image-text-to-video task, and we tried 3 existing methods to do this.

- [Make-it-move](https://github.com/Youncy-Hu/MAGE)
- [SuSIE](https://rail-berkeley.github.io/susie/)
- [Seer](https://seervideodiffusion.github.io/)

The working codes for these three methods are under the file [motion video data]. To run them, follow the readme under the corresponding files or follow the links above to go to their github repositoy.

However, all of them have issues of doing zero-shot tasks. Eventually, we use [Runway AI](https://runwayml.com/) and [KLing AI](https://klingai.com/) for this task. We have discussed methods on the report. The generated result can be found in the google drive https://drive.google.com/drive/folders/10VHhwZ4RnGIvJjm_OQ-MOkmij5cHy1Lv?usp=sharing. 

