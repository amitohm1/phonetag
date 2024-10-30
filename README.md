# phone tag
Phone App that will attempt to mimic "Laser Tag" by having each player use their phone as a "sight" and using the phone's camera images to distinguish between players.

Written in Kotlin, using Android Studios

## libraries used and references
1. CameraX: used for handling frames from camera.

https://medium.com/geekculture/getting-started-android-camerax-a84e138e2c00

I used the above link as starting code.

2. Tensorflowlite:

https://www.tensorflow.org/lite/examples/segmentation/overview

Current model I am using, there is also a writeup on posenet that can be navigated to, from here by clicking "pose estimation".

I used the ML pipe line from this example:

https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation/android/

The (lib_interpreter version) 


## points to work on to a working pitch demo

 - Implement using posenet (get added benefit of performance and allows
   us to differentiate shots to different parts of body).
	 - We also need to make sure that pre-trained posenet models exist that work with multiple instances of people (once we get to 3+ players).
 - We need to profile the code (I also didn't write the most efficient
   parallel code for iterating through the model's output matrix, I need
   to read through multi-threading notes and re-do this)
 - Once its profiled we can make improvement and get some FPS gains.
 - Looking into the scaling down and up of the camera images that are
   inputted to the model (the current model works on 257x257 images)
   Because I am using basic bitmap scaling provided, I believe the
   outline looks worse than it actually is. A possible solution to this
   is to make a 257x257 scope in the center and just crop the image at
   the exact model dimension (this should be computationally cheap).
 - Last goal: include more than 2 players allowed. I think this should
   be the last step and we should have a working version of 2 players
   first.

## ideas on how to move to multi player

 - Take video of each player rotating around, we generate images using
   the video and use that to reconstruct a 3D model of each player. We
   can then try using multiple computer vision solutions (eg: scaling
   the image then using a window of pixels to match feature points of
   players, or ML model)
 - Get the user to give us access to facebook/Instagram photos with
   their face labeled (this only helps us if the photo is from the
   front, so I don't know how robust this will be).
 - also note we could keep this computation off the phone and on a
   server that's connecting the players (if this ends up being a very
   expensive operation).

