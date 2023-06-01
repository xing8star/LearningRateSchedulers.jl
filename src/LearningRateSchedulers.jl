module LearningRateSchedulers
using Optimisers
export OptLrSheduler,
        lrscheduler!,
        LearningRateScheduler,
        resetscheduler!

mutable struct OptLrSheduler
    opt::Union{NamedTuple,Tuple}
    decay_iter::Vector
    lr::Float32
    current_lr_ratio::Int
    OptLrSheduler(opt::Union{NamedTuple,Tuple},decay_iter::Vector,lr::Float32)=new(opt,decay_iter,lr,1)
end

function lrscheduler!(opt::OptLrSheduler,current_epoch::Int)
    c=opt.current_lr_ratio
    if current_epoch<=1
        Optimisers.adjust!(opt.opt,opt.lr)
        opt.current_lr_ratio=1
    elseif c<=length(opt.decay_iter) && current_epoch>opt.decay_iter[c]
        opt.current_lr_ratio+=1
        Optimisers.adjust!(opt.opt,opt.lr*0.1^opt.current_lr_ratio)
    end
end

mutable struct LearningRateScheduler
    opt::Union{NamedTuple,Tuple}
    decay_iter::Vector
    lr::Float32
    current_lr_ratio::Int
    cnt::Int
    function OptLrSheduler(opt::Union{NamedTuple,Tuple},decay_iter::Vector,lr::Float32)
        new(opt,decay_iter,lr,1,1)
    end
end
function lrscheduler!(opt::LearningRateScheduler)
    c=opt.current_lr_ratio
    if c<=length(opt.decay_iter) && opt.cnt>opt.decay_iter[c]
        opt.current_lr_ratio+=1
        Optimisers.adjust!(opt.opt,opt.lr*0.1^opt.current_lr_ratio)
    end
end

function resetscheduler!(opt::LearningRateScheduler)
    Optimisers.adjust!(opt.opt,opt.lr)
    opt.current_lr_ratio=1
    opt.cnt=1
end

end # module SchedulerLR
