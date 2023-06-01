# LearningRateSchedulers.jl

A Scheduler adjusts lr for Optimisers.

# How to use

```julia
opt_schedule=LearningRateScheduler(opt,decay_iter,default_lr)
lrscheduler!(opt_schedule)
```