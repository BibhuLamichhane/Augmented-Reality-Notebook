# Air-Drawing

### My attempt on augmented reality with air writing. 

# How this works

- Uses the You Only Look Once(YOLO) model to find a hand in the frame. <br>
- Uses the keras retinanet model to find fingertips in an image of a hand. <br>
- Checks if the user is showing only an index finger.
- Coordinate of the index finger in each frame is stored.
- Lines are drawn based on the coordinates

# To draw

- Make sure our entire hand is in the frame <br>
- Raise only your index finger to start drawing <br>
- Raise all five of our fingers to clear the screen

<img src="https://github.com/BibhuLamichhane/Air-Writing/blob/master/AirWritng.gif"> <br>
(note : the video seems clunky because i had to convert it into a gif)
## If you want the weights send me a dm in one of my socials
- <a href="https://www.instagram.com/lamichhane_bibhu/">Instagram</a> <br>
- <a href="https://twitter.com/lamichhanebibhu">Twitter</a>

### Or email me at lamichhanebibhu0@gmail.com
