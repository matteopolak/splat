- get target image 512x512, load onto GPU (never changes, regular rgba texture)
- another texture to render into
- a buffer of the current random values
- a buffer of the best random values up to this point

loop:
- render image to texture (fragment shader)
- compare image to source (compute shader), write to another small buffer the difference
- store the "best difference" on CPU, load and compare. if it's better,
  copy the "current" buffer to the "best" buffer
- update random values on CPU by writing parts of the buffer in the command encoder1
