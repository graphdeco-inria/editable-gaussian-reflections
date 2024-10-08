TASKS

=============

FEATURES

- Get eval losses on glossy & diffuse images 

=============

BUGS

- The gaussian radius is incorrect in the compositing code 
- At 1k iters, it looks good, but after 5 or 10k, it collapses to all black (check for nans maybe?)

=============

LOW PRIO

- Make sure jointly training the diffuse works
- Nerf synthetic init for the dual scene only
- Faster camera loading for colmap also

=============


