# seam-carving
Original image: ![pano1](https://cloud.githubusercontent.com/assets/1958916/21788738/31bcae74-d685-11e6-9770-c81ea6ce1c63.jpg) 
Seam carving reduction of 300 pixels: ![pano1_reduced](https://cloud.githubusercontent.com/assets/1958916/21789868/056e62ec-d68d-11e6-8c2d-6755851a69f4.jpg)

This is a C++ implementation of the seam carving algorithm. The purpose of the seam carving algorithm is to rezise images without distorting the "important" sections of an image, as it would if you just normally tried to resize an image.

The seam carving algorithm computes a seam (an 8-connected path of pixels) from top-to-bottom or left-to-right. A seam is computed by traversing through a cumulative energy map of the image and choosing the path of least cost. In order to create a cumulative energy map, we need an energy image. To get an energy image, we use a gradient in the x and y directions of the image and then combine them to form an energy image:
![pano1_energy_image](https://cloud.githubusercontent.com/assets/1958916/21789696/dfa885d4-d68b-11e6-859a-71894c77b7ff.jpg)

From this energy image, we can calculate the cumulative energy map. We traverse in one direction (in this example, top-to-bottom). At each row, we loop through all the pixels. At each pixel, we examine the three upper pixels above the current one, choose the minimum of those,
and add those to the running total at that pixel. At the end, we form an image that can be represented as:
![pano1_cumulative_energy_map](https://cloud.githubusercontent.com/assets/1958916/21789695/dfa691c0-d68b-11e6-9ec3-1c142b7da647.jpg)
(blue indicates low energy, red indicates high energy)

From here, we look at the final row of the map, select the minimum value, and trace up the minimum path until the top of the image is reached. This becomes the seam that we remove. Rinse and repeat.

## More examples
Original:

![ocean](https://cloud.githubusercontent.com/assets/1958916/21791178/f934a906-d695-11e6-8166-052d6f35fab2.jpg) 

Resized using seam carving:

![ocean_result](https://cloud.githubusercontent.com/assets/1958916/21791177/f933f2cc-d695-11e6-833b-eb81243e44eb.jpg)
---
Original:

![painting](https://cloud.githubusercontent.com/assets/1958916/21791180/f9354014-d695-11e6-906f-30befeac87ba.jpg) 

Resized using seam carving:

![painting_result](https://cloud.githubusercontent.com/assets/1958916/21791179/f9350c2a-d695-11e6-9d2a-dd93a255d285.jpg)
---
Original:

![prague](https://cloud.githubusercontent.com/assets/1958916/21791181/f9354c80-d695-11e6-928a-c9188ba0ca14.jpg) 

Resized using seam carving:

![prague_result](https://cloud.githubusercontent.com/assets/1958916/21791182/f9362f2e-d695-11e6-8b99-186c9e476209.jpg)
