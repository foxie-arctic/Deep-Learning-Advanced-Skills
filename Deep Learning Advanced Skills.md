# Deep Learning Advanced Skills

## AutoDL

## Configuration

### argparse

argparse is a parser for command-line options, arguments and sub-commands. The argparse module makes it easy to write user-friendly command-line interfaces. The program defines what arguments it requires, and argparse will figure out how to parse those out of sys.argv. The argparse module also automatically generates help and usage messages. The module will also issue errors when users give the program invalid arguments.

3 core functions:

```
argparse.ArgumentParser
```

```
argparse.ArgumentParser.add_argument
```

```
argparse.ArgumentParser.parse_args
```

Basic Usage:

```
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()
print(args.accumulate(args.integers))
```

```
python prog.py -h
usage: prog.py [-h] [--sum] N [N ...]

Process some integers.

positional arguments:
 N           an integer for the accumulator

options:
 -h, --help  show this help message and exit
 --sum       sum the integers (default: find the max)
```

Page: [argparse](https://docs.python.org/3/library/argparse.html)

### configargparse

configargparse is a python package that allows users to configure their programs by a configuration file (`.ini` or `.yaml`).

To use configargparse, first you should install and import the package.

Then, you need to initialize a parser via `configargparse.ArgumentParser`, use `default_config_files` parameter in `__init__` to set the default path for config files. Or you may pass a config path from the terminal. Like `--config config/lego.txt`

Use `.add_argument()` to add argument to the initialized parser object.

For example, if this argument is the path for config file: set `is_config_file=True`.

Also, the following special values (whether in a config file or an environment variable) are handled in a special way to support booleans and lists:

* `key = true` is handled as if "`--key`" was specified on the command line. In your python code this key must be defined as a boolean flag (eg. `action="store_true"` or similar).

* `key = [value1, value2, ...]` is handled as if "`--key value1 --key value2`" etc. was specified on the command line. In your python code this key must be defined as a list (eg. `action="append"`).

Then, you need a configuration file. Here, we'll use [YAML syntax](https://learn.getgrav.org/17/advanced/yaml). But there are a variety of choices. 

Example (Python Code):
```
	import configargparse
	
	parser = configargparse.ArgumentParser()   
	 
	parser.add_argument('--config', is_config_file=True,
                        help='config file path')
	parser.add_argument("--expname", type=str,
                        help='experiment name')
	parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
	parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')
                        
    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024 * 32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024 * 64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=50000,
                        help='frequency of render_poses video saving')
                        
	args = parser.parse_args()
	
	config = args.config
	expname = args.expname
	basedir = args.basedir
	datadir = args.datadir
```

Example (`.txt`):
```
expname = blender_paper_chair
basedir = ./logs
datadir = ./data/nerf_synthetic/chair
dataset_type = blender

no_batching = True

use_viewdirs = True
white_bkgd = True
lrate_decay = 500

N_samples = 64
N_importance = 128
N_rand = 1024

precrop_iters = 500
precrop_frac = 0.5

half_res = True
```


Pages: 

[git repo for configargparse](https://github.com/bw2/ConfigArgParse)

[API for configargparse](https://bw2.github.io/ConfigArgParse/)

### OmegaConf

## Environment Variables on Linux

## Hugging Face

### Download pretrained models or parts of models

```
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, StableDiffusionPipeline

StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.precision_t)

DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=self.precision_t)
```

### `StableDiffusionPipeline`

#### Attention Processor for `StableDiffusionPipeline`

With an Attention Processor, one can do various of operations to every attention module in the UNet.

We can perform operations such as those mentioned in [Prompt-to-Prompt Image Editing with Cross Attention Control](https://arxiv.org/abs/2208.01626) leveraging Attention Processor. And you may also need to do some modifications to the standard `StableDiffusionPipeline` to perform some of the operations.

Generally, this is the workflow of a `StableDiffusionPipeline` with its `processor`:

```
    ddim_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    # can be other different schedulers
    
    SD = StableDiffusionPipeline.from_pretrained(
        model_id, scheduler=ddim_scheduler, torch_dtype=torch.float32,
        processor=CrossAttnCtrl(),
    ).to(device)
    # can be other modified StableDiffusionPipeline
    
    generator = torch.Generator(device).manual_seed(seed)
    # specify a random seed for the model, used in prompt2prompt

    images = SD(
        prompt=prompt, prompt_target=prompt_target,
        generator=generator, num_inference_steps=steps,
    ).images
    # generate images as the standard StableDiffusionPipeline does
    # modified StableDiffusionPipeline can have more self-defined parameters
    
```

There are 2 key steps.

Step 1: Generate a modified `StableDiffusionPipeline` object with your own Attention Processor object.

Step 2: Generate a result by calling your modified `StableDiffusionPipeline`.

The Attention Processor should be passed to the `UNet2DConditionModel` inside  your modified `StableDiffusionPipeline`. There is an Attention Processor for every `StableDiffusionPipeline`, for the default `StableDiffusionPipeline`, it is: `diffusers.models.attention_processor.AttnProcessor`.

(Note: If you use your modified pipeline, don't forget to add decorator `@torch.no_grad()` before `__call__` inside your pipeline.)

You should set your UNet's processor in the method of your pipeline with the help of `set_attn_processor` method of [`UNet2DConditionModel`](https://huggingface.co/docs/diffusers/api/models/unet2d-cond).


```
        if processor:
            self.processor = processor
            self.unet.set_attn_processor(self.processor)
```


Every time an attention module in the UNet is activated, the `forward` method of the UNet will use the `__call__` method from your Attention Processor to perform this part of the forward computing process.

So a Attention Processor should have the structure as:

```
->class MyAttnProcessor:
	->__call__()
	->... 
	# self-defined methods
```

You should follow the default version of Attention Processor(Pytorch < v2.0).

There are comments on how to modify it between the lines, so read it carefully!:

```
class AttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        # You may swap/save the attention_prob here.
        # To save them, you may use a dict whose keys are the id of the attn module. 
        # Or you may try __name__ method, but it operates on the class instead of an object directly since Python's objects have no name. 
        # You may use type(object) to get the class of the object. Then use __name__ to get a str, which can be a key for the dict.
        # But remember there will be multiple attention modules from the same class inside the UNet.
        # So I'll recommend you to use id(attn) as the keys.
        # Go check built-in function for id and dunder method for __name__ in Python Topic of this document.
        # So in the next round, you can get the attn map from the previous round for these attn modules correspondingly.
        # Then you can output the attention maps.
        
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
```

You should make sure that `__call__` has the following input parameters:

```
attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None,
```

Else you should go check ` AttnProcessor2_0`.

Set the inner variables of the processor inside the pipeline via directly access or self-defined methods of the processor before going through the UNet 
to do manipulation to the attention modules.(If you do it after the UNet part, then it will affect the next round instead of the current one.)

E.g:

```
self.processor.mapper, self.processor.alphas = get_seq_mapper(prompt_ids, prompt_target_ids, device)
# In StableDiffusionPipeline
```

Another common problem is that: how to pass parameters to the attention processor during runtime?

If you have a look at the declaration of `forward` function of `UNet2DConditionModel`, you'll find that there's a special parameter: `cross_attention_kwargs`

`cross_attention_kwargs (dict, optional)` : A kwargs dictionary that if specified is passed along to the AttnProcessor. 

Inside the code of `forward` function of `UNet2DConditionModel`, there're:

```

            sample, res_samples = downsample_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
                **additional_residuals,
            )

```

and

```
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )
```

and

```
             sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                upsample_size=upsample_size,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
             )
```

etc


Page: [Attention Processor](https://huggingface.co/docs/diffusers/api/attnprocessor)

#### Installation of `diffusers`

It is worth to notice that `diffusers` requires a certain version of `transformers` to work. If they are installed respectively using `pip`, it is very likely that bugs will appear.

In order to install `diffusers` correctly, you should go to [`diffusers` installation](https://huggingface.co/docs/diffusers/installation) to check.

If you are using Pytorch, it is recommended to install `diffusers` with the following:

```
pip install diffusers["torch"] transformers
```

#### LoRA fine-tuning/runtime-training method

In `StableDIffusionPipeline`, fine-tuning the pretrained diffusion model for some specific circumstances is a necessary skill. Although one can directly fine-tune UNet model by setting all the parameters inside it as requiring grad:

```
# Train mode
unet.train()
for p in unet.parameters():
	p.requires_grad_(True)
	
# Test mode
unet.eval()
for p in unet.parameters():
	p.requires_grad_(False)

```

However, once you finished the training process, if you want to reuse the fine-tuned model, you'll have to save the parameters of the entire UNet model. This takes much storage space. Another problem is that if you want to use both pretrained model and fine-tuned model at the same time, you'll have to load both of the UNet models, which will take a lot of VRAM. 

So in this part, we'll introduce how to use LoRA for fine-tuning(basically any kind of fine-tuning pipeline can use LoRA including DreamBooth, you can think LoRA as an additional part to UNet that enables flexible fine-tuning)

In `StableDiffusionPipeline`, LoRA is applied by using trainable attention processors.

First, by applying the following code:

```
            # Set correct lora layers
            self.lora_attn_procs = {}
            for name in self.unet.attn_processors.keys():
                cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
                if name.startswith("mid_block"):
                    hidden_size = self.unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = self.unet.config.block_out_channels[block_id]

                self.lora_attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                ).to(self.device)

            self.unet.set_attn_processor(self.lora_attn_procs)
            self.lora_layers = AttnProcsLayers(self.unet.attn_processors)
            self.lora_layers._load_state_dict_pre_hooks.clear()
            self.lora_layers._state_dict_hooks.clear()
            self.guidance_scale_lora = 1.0
```

In this case, to switch between train mode and test mode, you can use:

```
# Train mode
unet.train()
for p in self.lora_layers.parameters():
	p.requires_grad_(True)
	
# Test mode
unet.eval()
for p in self.lora_layers.parameters():
	p.requires_grad_(False)

```

To enable LoRA involving in the forward process of UNet, one should use `cross_attention_kwargs` to activate it. It will return the noise prediction using LoRA:

```
noise_pred_lora = unet(noisy_latents, t, encoder_hidden_states=embeddings,cross_attention_kwargs={'scale': 1.0}).sample                      
```

To deactivate LoRA in the forward process, one should also use `cross_attention_kwargs`. It will return the noise prediction using the pretrained  UNet:

```
noise_pred_lora = unet(noisy_latents, t, encoder_hidden_states=embeddings,cross_attention_kwargs={'scale': 0.0}).sample                      
```

In `forward` function of `UNet2DConditionModel`:

```
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0
```

Page: 

[`StableDiffusionPipeline`\-LoRA](https://huggingface.co/docs/diffusers/training/lora)

[ProlificDreamer Implementation of threestudio](https://github.com/threestudio-project/threestudio/blob/main/threestudio/models/guidance/stable_diffusion_vsd_guidance.py)

#### Memory saving in `StableDiffusionPipeline`

Page: [`StableDiffusionPipeline`](https://huggingface.co/docs/diffusers/v0.16.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline)

* `enable_attention_slicing`

Description: Enable sliced attention computation.

When to use: When this option is enabled, the attention module will split the input tensor in slices, to compute attention in several steps. This is useful to save some memory in exchange for a small speed decrease.

|Parameters|
|--|

|slice_size (str or int, optional, defaults to "auto")|
|--|
|When "auto", halves the input to the attention heads, so attention will be computed in two steps. If "max", maximum amount of memory will be saved by running only one slice at a time. If a number is provided, uses as many slices as attention_head_dim // slice_size. In this case, attention_head_dim must be a multiple of slice_size.| 


* `disable_attention_slicing`

Description: Contrary to `enable_attention_slicing`

When to use: When memory is not limited and willing to sacrifice space for speed.

* `enable_vae_slicing`

Description: Enable sliced VAE decoding computation.

When to use: When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.

* `disable_vae_slicing`

Description: Contrary to `enable_vae_slicing`

When to use: When memory is not limited and willing to sacrifice space for speed.

* `enable_model_cpu_offload`

Description: Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its forward method is called, and the model remains in GPU until the next model runs.

When to use: Save some memory with performance preserved.

* `enable_sequential_cpu_offload`

Description: Offloads all models to CPU using accelerate, significantly reducing memory usage. When `called`, `unet`, `text_encoder`, `vae` and `safety checker` have their state dicts saved to CPU and then are moved to a `torch.device('meta')` and loaded to GPU only when their specific submodule has its `forward` method called.

When to use: Save more memory but lose speed.

* `self.unet.to(memory_format=torch.channels_last)`

* (optional)`enable_xformers_memory_efficient_attention`
* (optional)`disable_xformers_memory_efficient_attention`

## Python

### Built-in Methods

* `getattr`: retrieve the value of an attribute from an object using its name as a string

In Python, the `getattr()` function is a built-in function that allows you to retrieve the value of an attribute from an object using its name as a string. It's particularly useful when you want to access an attribute dynamically, meaning that you only know the attribute's name as a string at runtime. `getattr()` is often used for introspection, dynamic attribute access, and to handle cases where the attribute might not exist.

The general syntax of the `getattr()` function is as follows:

```python
getattr(object, attribute_name[, default])
```

- `object`: The object from which you want to retrieve the attribute.
- `attribute_name`: A string representing the name of the attribute you want to access.
- `default` (optional): A value to be returned if the attribute doesn't exist. If not provided and the attribute doesn't exist, `getattr()` will raise an `AttributeError`.

Here's an example that demonstrates how `getattr()` works:

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

person = Person("Alice", 30)

# Using getattr() to dynamically access attributes
attribute_name = "name"
name = getattr(person, attribute_name)
print(name)  # Output: Alice

attribute_name = "age"
age = getattr(person, attribute_name)
print(age)   # Output: 30

# Trying to access a non-existent attribute
nonexistent_attribute = "location"
default_value = "Unknown"
location = getattr(person, nonexistent_attribute, default_value)
print(location)  # Output: Unknown
```

In this example, `getattr()` is used to dynamically access the `name` and `age` attributes of the `person` object. It's also used to access a non-existent attribute, `location`, with a default value of `"Unknown"`.

The `getattr()` function is particularly handy when you're working with objects whose attribute names are determined at runtime or when you're designing code that involves introspection, metaprogramming, or dynamic configuration.

* `id`: returns the unique identity (memory address) of an object

In Python, `id()` is a built-in function that returns the unique identity (memory address) of an object. This identity is typically represented as an integer, and it remains constant for the lifetime of the object. It's important to note that `id()` does not reveal any details about the object's content or type; it solely provides a way to differentiate between different objects based on their memory locations.

The `id()` function takes a single argument, which is the object for which you want to retrieve the identity. Here's the basic syntax:

```python
object_id = id(object)
```

Here's a simple example to demonstrate how `id()` works:

```python
x = 42
y = x

print(id(x))  # Prints the memory address of x
print(id(y))  # Prints the memory address of y (same as x)
```

In this example, both `x` and `y` point to the same memory location because integers are immutable in Python. When you assign `y = x`, you're not creating a new integer object; you're simply creating another reference (`y`) to the same object that `x` refers to. Therefore, the `id()` of both variables will be the same.

On the other hand, if you were to create a mutable object, such as a list, and modify it, the `id()` of the object would remain the same even though its contents have changed. This is because the identity of the object itself hasn't changed, only its internal state.

```python
list1 = [1, 2, 3]
list2 = list1

print(id(list1))  # Prints the memory address of list1
print(id(list2))  # Prints the memory address of list2 (same as list1)

list1.append(4)
print(list1)      # Prints [1, 2, 3, 4]
print(list2)      # Also prints [1, 2, 3, 4], since list2 and list1 refer to the same list object
```

In summary, the `id()` function is useful for understanding memory management and object references in Python. It's a way to differentiate between different objects based on their unique memory addresses, allowing you to identify whether two variables are referring to the same underlying object or not.

(generated by ChatGPT)

* `sorted`: returns a new sorted list containing the elements from the original collection in the desired order, without modifying the original collection

In Python, the `sorted()` function is a built-in function that is used to sort a collection of items, such as a list, tuple, or any iterable, in a specified order. It returns a new sorted list containing the elements from the original collection in the desired order, without modifying the original collection.

The `sorted()` function can take multiple arguments, but the primary argument is the iterable you want to sort. It can also take optional arguments to customize the sorting behavior:

```python
# Sorting a list of numbers in ascending order
numbers = [4, 2, 8, 5, 1, 9]
sorted_numbers = sorted(numbers)
print(sorted_numbers)  # Output: [1, 2, 4, 5, 8, 9]

# Sorting a list of strings in alphabetical order
fruits = ["apple", "banana", "cherry", "date"]
sorted_fruits = sorted(fruits)
print(sorted_fruits)  # Output: ['apple', 'banana', 'cherry', 'date']
```

You can also use the `key` parameter to specify a custom sorting key, which is a function that generates a value based on each element. The sorting is then done based on these generated values:

```python
# Sorting a list of strings by their lengths
words = ["apple", "banana", "cherry", "date"]
sorted_words = sorted(words, key=len)
print(sorted_words)  # Output: ['date', 'apple', 'cherry', 'banana']
```

Additionally, the `reverse` parameter can be used to sort in descending order:

```python
# Sorting a list of numbers in descending order
numbers = [4, 2, 8, 5, 1, 9]
sorted_numbers_desc = sorted(numbers, reverse=True)
print(sorted_numbers_desc)  # Output: [9, 8, 5, 4, 2, 1]
```

It's important to note that the `sorted()` function returns a new sorted list and doesn't modify the original list. If you want to sort a list in-place (i.e., modify the original list), you can use the `list.sort()` method:

```python
numbers = [4, 2, 8, 5, 1, 9]
numbers.sort()  # Sorts the list in-place
print(numbers)  # Output: [1, 2, 4, 5, 8, 9]
```

The `sorted()` function in Python can be used to sort a dictionary, but it doesn't directly sort the dictionary items. Instead, it sorts the keys of the dictionary (or the keys based on a custom key function) and returns a list of sorted keys. You can then use these sorted keys to access the corresponding values from the original dictionary.

Here's how you can use `sorted()` to sort a dictionary based on its keys:

```python
my_dict = {'banana': 3, 'apple': 1, 'pear': 2, 'orange': 4}

sorted_keys = sorted(my_dict.keys())  # Sorting dictionary keys
sorted_dict = {key: my_dict[key] for key in sorted_keys}

print(sorted_dict)
```

Output:
```
{'apple': 1, 'banana': 3, 'orange': 4, 'pear': 2}
```

In this example, the `sorted()` function is used to sort the keys of the `my_dict` dictionary in ascending order. Then, a new dictionary `sorted_dict` is created by iterating through the sorted keys and extracting corresponding values from the original dictionary.

If you want to sort the dictionary items based on their values, you can use the `key` parameter of the `sorted()` function to provide a custom sorting key that considers the values. For instance:

```python
my_dict = {'banana': 3, 'apple': 1, 'pear': 2, 'orange': 4}

sorted_items = sorted(my_dict.items(), key=lambda item: item[1])  # Sorting by values
sorted_dict = dict(sorted_items)

print(sorted_dict)
```

Output:
```
{'apple': 1, 'pear': 2, 'banana': 3, 'orange': 4}
```

In this example, the `sorted()` function is used to sort the dictionary items based on their values. The `key` parameter specifies a lambda function that extracts the second element of each tuple (i.e., the value) to use as the sorting key.

Remember that dictionaries in Python are inherently unordered collections prior to Python 3.7. Starting from Python 3.7, dictionaries maintain the insertion order. However, if you need to work with dictionary-like data that maintains order, you might consider using the `collections.OrderedDict` class.

In summary, the `sorted()` function is a versatile tool in Python for sorting iterable objects in various ways, providing control over sorting criteria and order while preserving the original data.

(generated by ChatGPT)

* `vars`: returns the `__dict__` attribute of an object

In Python, the `vars()` function is a built-in function that returns the `__dict__` attribute of an object. The `__dict__` attribute is a dictionary that contains the object's attributes and their corresponding values. This function is commonly used with instances of classes to retrieve their attributes as a dictionary.

Here's how the `vars()` function works:

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

person = Person("Alice", 30)

# Using vars() to retrieve attributes of the object
attributes = vars(person)
print(attributes)
```

Output:
```
{'name': 'Alice', 'age': 30}
```

In this example, the `vars()` function is used to obtain the attributes and their values of the `person` object. The output is a dictionary where keys are attribute names ("name" and "age" in this case), and values are the corresponding attribute values.

It's important to note that `vars()` works only with objects that have a `__dict__` attribute. This includes instances of most user-defined classes. However, built-in objects and some specialized objects might not have a `__dict__` attribute, in which case calling `vars()` on them will raise an exception.

Keep in mind that while `vars()` is a useful tool for introspection and debugging, directly accessing an object's attributes through its `__dict__` or `vars()` might not be the most idiomatic way to interact with objects in Python. Instead, it's often recommended to use regular attribute access (`object.attribute`) to access and manipulate an object's attributes.

(generated by ChatGPT)

### contextlib

This module provides utilities for common tasks involving the with statement. For more information see also [Context Manager Types](https://docs.python.org/3/library/stdtypes.html#typecontextmanager) and [With Statement Context Managers](https://docs.python.org/3/reference/datamodel.html#context-managers).

Generally, a with context manager needs to have 2 functions declared: `__enter__` and `__exit__`.

* `@contextmanager`

This function is a decorator that can be used to define a factory function for with statement context managers, without needing to create a class or separate `__enter__()` and `__exit__()` methods.

While many objects natively support use in with statements, sometimes a resource needs to be managed that isn’t a context manager in its own right, and doesn’t implement a `close()` method for use with contextlib.closing

An abstract example would be the following to ensure correct resource management:

```
from contextlib import contextmanager

@contextmanager
def managed_resource(*args, **kwds):
    # Code to acquire resource, e.g.:
    resource = acquire_resource(*args, **kwds)
    try:
        yield resource
    finally:
        # Code to release resource, e.g.:
        release_resource(resource)
```

The function can then be used like this:

```
with managed_resource(timeout=3600) as resource:

    # Resource is released at the end of this block,

    # even if code in the block raises an exception
```

The function being decorated must return a generator-iterator when called. This iterator must yield exactly one value, which will be bound to the targets in the with statement’s as clause, if any.

At the point where the generator yields, the block nested in the with statement is executed. The generator is then resumed after the block is exited. If an unhandled exception occurs in the block, it is reraised inside the generator at the point where the yield occurred. Thus, you can use a `try…except…finally` statement to trap the error (if any), or ensure that some cleanup takes place. If an exception is trapped merely in order to log it or to perform some action (rather than to suppress it entirely), the generator must reraise that exception. Otherwise the generator context manager will indicate to the with statement that the exception has been handled, and execution will resume with the statement immediately following the with statement.

`contextmanager()` uses `ContextDecorator` so the context managers it creates can be used as decorators as well as in with statements. When used as a decorator, a new generator instance is implicitly created on each function call (this allows the otherwise “one-shot” context managers created by `contextmanager()` to meet the requirement that context managers support multiple invocations in order to be used as decorators).

An example of using `@contextmanager` to disable `class_embedding` in `unet` within the block of with statement:

```
    @contextmanager
    def disable_unet_class_embedding(self):
        class_embedding = self.unet.class_embedding
        try:
            self.unet.class_embedding = None
            yield self.unet
        finally:
            self.unet.class_embedding = class_embedding
```

```
    with guidance.disable_unet_class_embedding() as unet:
    		noise_pred_pretrain = unet(latent_model_input, tt, encoder_hidden_states=text_embeddings,cross_attention_kwargs={"scale": 0.0}).sample
```





Page: [contextlib](https://docs.python.org/3/library/contextlib.html)

### Decorators

* `@dataclass`: Similar to `struct` in `C++`

Page: [`@dataclass` on Zhihu](https://zhuanlan.zhihu.com/p/419778289)

### Dunder(double underscore) Methods

In Python, attributes or methods that have names starting and ending with double underscores, like `.__name__`, are known as "dunder" (double underscore) methods or magic methods. These dunder methods have a special significance in the language and are used to provide specific behavior for objects when certain operations or actions are performed on them.

The use of double underscores is a convention to make these special methods stand out from regular user-defined attributes or methods. Python uses this naming convention to avoid clashes with user-defined names and to clearly indicate that these methods have a reserved purpose.

For example, `.__name__` is a dunder method used to get the name of a function, class, or module. If Python used a simple name like `.name` without the double underscores, it could potentially clash with a user-defined attribute or method named `name` on a class or module.

By using `.__name__`, Python ensures that the name is reserved for a specific purpose and won't accidentally interfere with user-defined attributes or methods. It's a way of making the language more robust and consistent.

Here are a few more examples of common dunder methods in Python:

- `.__init__`: The constructor method used to initialize an object.
- `.__str__`: The method that returns the string representation of an object when `str()` function is called on it.
- `.__add__`: The method used for addition when `+` operator is used between two objects.
- `.__len__`: The method used to get the length of an object when `len()` function is called on it.

Using double underscores as a naming convention for special methods is just one aspect of Python's philosophy of "explicit is better than implicit" and "there should be one-- and preferably only one --obvious way to do it."

(generated by ChatGPT)

* `.__name__`

In Python, `.__name__` is a special attribute that is commonly used with functions, classes, and modules. It allows you to access the name of the object it is attached to. The value of `.__name__` will vary depending on the type of object.

1. Function `.__name__`:
When used with a function, `.__name__` returns the name of the function as a string. For example:

```python
def my_function():
    pass

print(my_function.__name__)  # Output: "my_function"
```

2. Class `.__name__`:
When used with a class, `.__name__` returns the name of the class as a string. For example:

```python
class MyClass:
    pass

print(MyClass.__name__)  # Output: "MyClass"
```

3. Module `.__name__`:
When used with a module, `.__name__` returns the name of the module as a string. For example, consider a module named `my_module.py`:

```python
# my_module.py
def some_function():
    pass

print(__name__)  # Output: "__main__" when executed directly

# When the module is imported in another script
# Output: "my_module" when imported in another script
```

Note that if the module is executed directly (i.e., it's the main script), `__name__` will be set to `"__main__"`. If it's imported into another script, `__name__` will have the value of the module's actual name, such as `"my_module"` in the example above.

The `.__name__` attribute is particularly useful when writing code that can be executed both as a standalone script and as a module to be imported into other scripts. It allows you to check whether the code is being executed directly or imported, and this can help you define specific behavior in each case.

### Generator

Generators in Python are a special type of iterable, and they are a powerful feature for working with sequences of data. Generators allow you to create iterable sequences on-the-fly, producing values one at a time rather than loading the entire sequence into memory. They are defined using functions and the `yield` keyword.

Generators use lazy evaluation, which means they produce values one at a time as requested. The function's execution is paused and resumed at each `yield` statement. This is efficient for working with large or infinite sequences.

* cycle:

```
    def cycle(iterable):
        iterator = iter(iterable)
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                iterator = iter(iterable)
```

1. It starts by creating an iterator from the input iterable using the `iter()` function.

2. It enters an infinite loop with `while True`.

3. Inside the loop, it tries to yield the next element from the iterator using `yield next(iterator)`.

4. If the iterator is exhausted and raises a `StopIteration` exception (which is a common way iterators signal the end), it catches the exception and resets the iterator by creating a new one with `iterator = iter(iterable)`. This effectively restarts the iteration from the beginning of the input iterable, allowing it to cycle through the elements again.


## PyTorch

### Channel Last Format

Page: [Channel Last Format in Pytorch](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)

Description: Channels last memory format is an alternative way of ordering NCHW tensors in memory preserving dimensions ordering. Channels last tensors ordered in such a way that channels become the densest dimension (aka storing images pixel-per-pixel).

APIs:

Classical Continuous Tensor
```
import torch

N, C, H, W = 10, 3, 32, 32
x = torch.empty(N, C, H, W)
print(x.stride())  # Outputs: (3072, 1024, 32, 1)
```

```
(3072, 1024, 32, 1)
```

Conversion Operator
```
x = x.to(memory_format=torch.channels_last)
print(x.shape)  # Outputs: (10, 3, 32, 32) as dimensions order preserved
print(x.stride())  # Outputs: (3072, 1, 96, 3)
```

```
torch.Size([10, 3, 32, 32])
(3072, 1, 96, 3)
```

Back to contiguous

```
x = x.to(memory_format=torch.contiguous_format)
print(x.stride())  # Outputs: (3072, 1024, 32, 1)
```

```
(3072, 1024, 32, 1)
```

Alternative Operation

```
x = x.contiguous(memory_format=torch.channels_last)
print(x.stride())  # Outputs: (3072, 1, 96, 3)
```

```
(3072, 1, 96, 3)
```

Format Checks

```
print(x.is_contiguous(memory_format=torch.channels_last))  # Outputs: True
```

```
True
```

### DataLoader

#### Define a loader that provides data for you.

Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.

The DataLoader supports both map-style and iterable-style datasets with single- or multi-process loading, customizing loading order and optional automatic batching (collation) and memory pinning.

```
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
``` 

Pages: 

[`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader)

[`torch.utils.data`](https://pytorch.org/docs/stable/data.html?highlight=dataloader#module-torch.utils.data)

[`collate_fn`](https://zhuanlan.zhihu.com/p/493400057)

### Datatype

#### Define your own datatype.

```
@dataclass
class UNet2DConditionOutput:
    sample: torch.HalfTensor
```

### EMA


### Gradient & Back-Propagation

#### Extending `torch.autograd` on `Function`

Description: Implement a custom function on your own for `torch.autograd`. 

When to use: Implement not differentiable or non-`Pytorch`(e.g `numpy`) version or use `C++` extension or self-defined workflow such as SDS in DreamFusion, but still wish for your operation to chain with other ops and work with the autograd engine.

When not to use: It is not implemented for circumstances if you want to alter gradients during the backward process. Refer to `tensor` or `Module` hook for this part. It should not be used if you want to maintain state. Check for extending `torch.nn`.

Pages:
 
[Extending `torch.autograd`](http://pytorch.org/docs/stable/notes/extending.html#extending-autograd)

[`torch.autograd.Function` in amp](https://pytorch.org/docs/stable/notes/amp_examples.html)

[`torch.autograd.Function`](https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function)

```
class MyMM(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a.mm(b)
    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        return grad.mm(b.t()), a.t().mm(grad)

mymm = MyMM.apply

with autocast(device_type='cuda', dtype=torch.float16):
    output = mymm(input1, input2)
```

Warning: Use apply to initialize the Function first, do not use forward method directly.

### Just-In-Time 

### Mixed=Precision Training

### Muti-GPU Training

Pages:

[DDP-1](https://zhuanlan.zhihu.com/p/178402798)

[DDP-2](https://zhuanlan.zhihu.com/p/187610959)

[DDP-3](https://zhuanlan.zhihu.com/p/250471767)

### Network

#### Adding `torch.Tensor` to model's parameters

`Parameter` is a kind of Tensor that is to be considered a module parameter.

```
torch.nn.parameter.Parameter(data=None, requires_grad=True)
```

`Parameter`s are `Tensor` subclasses, that have a very special property when used with `Module`s - when they’re assigned as `Module` attributes they are automatically added to the list of its parameters, and will appear e.g. in `parameters()` iterator. Assigning a `Tensor` doesn’t have such effect. This is because one might want to cache some temporary state, like last hidden state of the RNN, in the model. If there was no such class as `Parameter`, these temporaries would get registered too.

So basically, there are 2 steps to add a `Tensor` to model's parameters.

1. Wrap the `Tensor` up with `torch.nn.parameter.Parameter` and determine whether it needs to be updated during training.

2. Assign the new `Parameter` as one of the attributes of a `Module`, then it will be automatically added to the `module`'s parameters list.  

Or, if you don't want the `Parameter` to be the attribute of a `Module`, you may directly pass it to the `optimizer`



Pages: 

[`torch.nn.parameter.Parameter`](https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter)

#### Frozen parameters in training

#### Iterating through all sub-modules in a `torch.nn.Module` object via `.named_modules()`

In PyTorch, `.named_modules()` is a method available for neural network modules, such as `torch.nn.Module`, which is the base class for all PyTorch neural network modules. This method is used to get an iterator over all the sub-modules (child modules) within the current module and their names.

Here's how it works:

1. For a given PyTorch `Module`, you can call `.named_modules()` to get an iterator that yields pairs of (name, module) for each sub-module contained within the current module.

2. The "name" refers to the name of the sub-module within the parent module, and "module" refers to the sub-module object itself.

3. The iterator traverses the module and all its sub-modules in a depth-first manner, meaning it first visits the current module, then its children, and so on.

Here's an example of how you can use `.named_modules()` in PyTorch:

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(10, 5)
        self.layer2 = nn.Linear(5, 3)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

model = MyModel()

# Iterating over named modules and printing their names and modules
for name, module in model.named_modules():
    print(f"Name: {name}, Module: {module}")
```

Output:
```
Name: , Module: MyModel(
  (layer1): Linear(in_features=10, out_features=5, bias=True)
  (layer2): Linear(in_features=5, out_features=3, bias=True)
)
Name: layer1, Module: Linear(in_features=10, out_features=5, bias=True)
Name: layer2, Module: Linear(in_features=5, out_features=3, bias=True)
```

As you can see, the iterator includes the top-level module (`MyModel`) and its sub-modules (`layer1` and `layer2`), along with their corresponding objects. This can be helpful, for example, when you want to inspect or modify specific layers within a complex neural network architecture.

(generated by ChatGPT, checked)

Page:[`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Page: [`torch.nn`](https://pytorch.org/docs/stable/nn.html)

### Optimizer & Scheduler

Page: [`torch.optim`](https://pytorch.org/docs/stable/optim.html)

#### `capturable` parameter in `Adam` and `AdamW` since Pytorch v1.12.0 

After Pytorch v1.12.0, a new parameter is added to `Adam` and `AdamW` optimizers.

It may cause troubles when you are training a repo built on Pytorch versions lower than v1.12.0 with your local environment using Pytorch versions higher than v1.12.0+. 

You may see:

```
AssertionError: If capturable=False, state_steps should not be CUDA tensors.
```

or

```
AssertionError: If capturable=True, params and state_steps must be CUDA tensors.
```

When this happens, set the `capturable` parameter as its opposite value:

```
    # For Pytorch >= 1.12.0, set capturable = True for Adam & AdamW
    for param_group in optimizer.param_groups:
        param_group['capturable'] = True   # Or False
```



### Scaler

### Seed

#### `torch.manual_seed` & `torch.cuda.manual_seed`

Description: Set a certain random seed in torch.

#### `seed_everything`

```
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True
```

1. `torch.manual_seed(seed)`: This line sets the seed for the random number generator used by PyTorch on the CPU. By setting this seed, you ensure that any random operations performed on the CPU will produce the same results every time the code is run with the same seed. This is important for reproducibility because machine learning models often involve randomness in their initialization and training processes.

2. `torch.cuda.manual_seed(seed)`: This line sets the seed for the random number generator used by PyTorch on the GPU (if a GPU is available and PyTorch is configured to use it). Just like the previous line, this ensures that random operations on the GPU will be reproducible when the code is run with the same seed. It's crucial to set both CPU and GPU seeds if your code involves GPU computations.

3. `#torch.backends.cudnn.deterministic = True`: This line is commented out with a `#`. In PyTorch, the CuDNN (CUDA Deep Neural Network library) is used for GPU-accelerated deep learning operations. If you uncomment this line and set it to `True`, it forces CuDNN to use deterministic algorithms for certain operations, which can further enhance reproducibility. However, it may come at the cost of performance in some cases, so it's often left as an option to enable or disable depending on your specific needs.

4. `#torch.backends.cudnn.benchmark = True`: Similar to the previous line, this one is also commented out. If you uncomment it and set it to `True`, it enables CuDNN benchmarking mode, which can optimize performance for your specific GPU hardware. However, enabling this mode may result in non-reproducible results, as it can make use of GPU-specific optimizations that introduce variability. So, it's usually disabled for reproducibility purposes.

(generated by ChatGPT)

```
def seed_everything(seed, local_rank):
    random.seed(seed + local_rank)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed + local_rank)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.benchmark = True  
```

In distributed computing environments, ensuring reproducibility and consistency of results is crucial, especially when multiple computing nodes are involved simultaneously. The presence of randomness can lead to inconsistent results across nodes. To address this, setting random seeds is necessary to control the randomness.

This function is designed to set random seeds to ensure consistent randomness across different computing nodes. Let's break down each line:

    random.seed(seed + local_rank): Sets the seed for Python's random number generator. By adding the global seed seed to the local_rank, this ensures that each process in the distributed environment has a unique but reproducible random seed.

    os.environ['PYTHONHASHSEED'] = str(seed): Sets the hash seed for Python. This controls the randomness of operations that use hash functions, ensuring consistency across different nodes.

    np.random.seed(seed + local_rank): Sets the seed for NumPy's random number generator, similar to Python's random number generator. This ensures consistent randomness across different nodes when using NumPy.

    torch.manual_seed(seed): Sets the seed for PyTorch's random number generator. This ensures consistent randomness when performing computations using PyTorch.

    torch.cuda.manual_seed(seed): Sets the seed for PyTorch's GPU random number generator, ensuring consistent randomness when utilizing GPUs for computations.

    # torch.backends.cudnn.benchmark = True: This line is commented out, but it appears to enable PyTorch's cuDNN benchmark mode. This mode can optimize performance for convolutional neural networks on compatible GPUs by dynamically selecting the best algorithms for convolutions.

In summary, this function sets various random number generator seeds to maintain consistent randomness across different nodes in a distributed computing environment, enhancing the reproducibility and stability of experiments. The commented line suggests an optimization option related to GPU computations in PyTorch.

(generated with ChatGPT)

## PyTorch Lightning

## Tensorboard

Pages: 

[Tensorboard](https://www.tensorflow.org/tensorboard/)

[To use Tensorboard with PyTorch](https://pytorch.org/docs/1.12/tensorboard.html?highlight=tensorboard#module-torch.utils.tensorboard)

## Trimesh
