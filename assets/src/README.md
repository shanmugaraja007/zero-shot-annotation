# Diagram sources

`diagram.html` is the animated pipeline figure. Every number shown in it is read
off the defaults in `../../configs/` and `../../src/zsa/config.py`; if those
change, update the figure.

Rendering:

```bash
python shoot.py 120                      # writes frames/
ffmpeg -framerate 20 -i frames/f%04d.png -c:v libx264 -pix_fmt yuv420p -crf 20 ../pipeline.mp4
ffmpeg -framerate 20 -i frames/f%04d.png -vf "scale=1100:-1,palettegen=max_colors=128:stats_mode=diff" pal.png
ffmpeg -framerate 20 -i frames/f%04d.png -i pal.png \
  -lavfi "scale=1100:-1[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4" ../pipeline.gif
```

## Image provenance

| File | What it is |
| --- | --- |
| `f_input.jpg` | Off road track, the unlabelled input frame. |
| `f_output.jpg` | The matching semantic / index mask (sky, vegetation, grass, track). |
