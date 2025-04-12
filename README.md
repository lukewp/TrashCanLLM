# TrashCanLLM
Where I'm keeping notes and relevant code on how to use a 2013 Mac Pro (MacPro6,1) in modern times to drive LLMs.

## The system:
I'm using a 2013 Mac Pro (a "Trash Can") with a 2.7Ghz Xeon E5-2697 V2 12-core CPU, 64GB RAM, and dual AMD FirePro D700 6GB VRAM GPUs. I have replaced the SSD with an NVMe but it's otherwise as it rolled out of the Apple Factory in 2014.

## What it's doing:
I acquired this as an experiment, with some hope that I could get it running to support Large Language Models (LLMs). I am not optimizing for time or energy consumption, but I do want to take advantage of large token context windows and run relatively large models, ideally in parallel.
A couple of things I'm actively using it for:
* Running multiple simultaneous llama-server instances of smaller (<=8B parameter) models with different settings to support different tasks without having to deal with loading and unloading models to switch from task to task, with one set of layers (prompt processing) in GPU/VRAM and everything else in CPU/RAM.
* Running one big (Llama-4 Scout 4-bit quantization) LLM with a large context window
* Can run all major code completion models via VPN as the back-end to Cursor, Zed, Windsurf, etc when you run out of Claude tokens or don't want to be throttled.

## Why am I doing this?
I bought the Mac Pro for about $400. It has 12gb of VRAM, 64 GB of RAM (upgradeable to 128), and a 12-core CPU. It is not anywhere near the fastest computer one can buy in 2025 but it will run models that paralyze much more expensive, faster machines. Tuned correctly, the combined 76 GB of (V)RAM can pack a massive punch. HOWEVER, it is not efficient from a Watt-Hours-per-token perspective. But unless you're using it all the time, it's still cheaper than a monthly subscription to any of the major cloud-based LLMs and doesn't create privacy concerns.

## What are the challenges?
Mainly getting the GPUs to engage has been a major challenge. They primarily help in prompt analysis (so speeding up RAG tasks, for example) but don't seem to make a huge impact on token generation rates in text completion. But I beat my head against a wall for a week trying to get both Windows and Linux to pick up the FirePros and was unsuccessful. MacOS was the only OS that would recognize and engage them. BUT, you have to downgrade to an older version of MacOS (Monterey). And as of the time of this writing, you have to do some driver manipulation and patching to get Vulkan and MoltenVK compiled in a way that'll enable the MacOS-friendly LLM hosts to run them. Even though they're both llama.cpp wrappers, I have not gotten LM-Studio to work at all. And Ollama won't pick up the GPUs.

## OK so how do we do this?
Here's the general steps:
1. Use OpenCore Legacy Patcher to install MacOS Monterey version 12.7.4 (https://dortania.github.io/OpenCore-Legacy-Patcher/)
1. Read these two things for background -- my solution is largely a blend of the two: https://medium.com/@ankitbabber/run-a-llm-locally-on-an-intel-mac-with-an-egpu-55ed66db54be and https://github.com/ollama/ollama/issues/1016#issuecomment-2642713162
2. install homebrew (if not already installed) (instructions here: https://brew.sh/)
3. in terminal $ `brew install cmake libomp cmake-docs`
4. download and install Xcode 14.2 from Mac Developer Hub (https://developer.apple.com/download/all/) -- you have to log in with your Apple ID. (note, Xcode Command Line Tools is not good enough)
5. run Xcode, and in Settings look for a "command line tools location" option and point it at your full XCode installation
6. Install the latest version of Vulkan SDK: https://vulkan.lunarg.com/sdk/home
7. Reboot your computer
8. Run the Vulkan Configurator and leave it open. I set everything to "Auto" and unchecked all the logging boxes to try to improve performance/lessen verbosity but more info might help identify and fix bugs
9. Open a single terminal window and only use that for all the following (because environment variables are important and not global across terminal sessions / outside terminal sessions):
10. From VulkanSDK directory (wherever you installed it): $ `source ./setup-env.sh`
11. Type: $ `export LDFLAGS="-L/usr/local/opt/libomp/lib"` and $ `export CPPFLAGS="-I/usr/local/opt/libomp/include"`
12. Build a patched version of MoltenVK:
    ```
    # Set up repo:
    git clone https://github.com/KhronosGroup/MoltenVK.git
    cd MoltenVK
    git fetch origin pull/2434/head:p2434
    git switch p2434

    # Compile & Install:
    ./fetchDependencies --macos
    make macos
    make install
    ```
13. Build llama.cpp with a set of custom compilation flags:
    ```
    # Set up repo:
    git clone https://github.com/ggml-org/llama.cpp.git
    cd llama.cpp
    
    # Test vulkan from the command line to ensure llama will find drivers:
    vulkaninfo
    vkvia

    # Build llama with Vulkan and OpenMP:
    cmake -B build -DGGML_METAL=OFF -DGGML_VULKAN=ON \
    -DOpenMP_C_FLAGS=-fopenmp=lomp \
    -DOpenMP_CXX_FLAGS=-fopenmp=lomp \
    -DOpenMP_C_LIB_NAMES="libomp" \
    -DOpenMP_CXX_LIB_NAMES="libomp" \
    -DOpenMP_libomp_LIBRARY="$(brew --prefix)/opt/libomp/lib/libomp.dylib" \
    -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp $(brew --prefix)/opt/libomp/lib/libomp.dylib -I$(brew --prefix)/opt/libomp/include" \
    -DOpenMP_CXX_LIB_NAMES="libomp" \
    -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp $(brew --prefix)/opt/libomp/lib/libomp.dylib -I$(brew --prefix)/opt/libomp/include"

    # Compile:
    cmake --build build --config Release
    ```
14. Watch the compilation process. It will throw a bunch of warnings around 6-7% and 11%.
15. Assuming it's complete, you need to download a model (I throw them in a separate directory outside the llama folder called ../llm-models) and test it via CLI:
    ```
    mkdir ../llm-models
    cd ../llm-models
    # Download the 8-bit quantization of IBM Granite's 2B parameter Granite model (about 2.7 gigs)
    wget https://huggingface.co/lmstudio-community/granite-3.2-2b-instruct-GGUF/resolve/main/granite-3.2-2b-instruct-Q8_0.gguf?download=true

    # Go back to your Llama build and run it:
    cd ../llama.cpp
    ./build/bin/llama-cli -m ../llm-models/granite-3.2-2b-instruct-Q8_0.gguf
    ```
    With no parameters, pop open your Activity Monitor and see whether this is running on your CPU or GPUs. Have a brief conversation with it via the CLI and see whether it's giving you rational and complete responses.
16. It may not load up your GPUs by default. You can instruct it to load layers into your GPUs via the -ngl flag:
    ```
    ./build/bin/llama-cli -m ../llm-models/granite-3.2-2b-instruct-Q8_0.gguf -ngl 99
    ```
    Give this a prompt via the CLI and then check and see whether it's showing up any different in your Activity Monitor.
17. Load the model and serve via web interface to your local network:
    ```
    ./build/bin/llama-server -m ../llm-models/granite-3.2-2b-instruct-Q8_0.gguf -ngl 99 --host :: --port 8080
    ```
    You should be able to access from that machine using http://localhost:8080 and from any other machine on the local network by hitting its internal IP address with :8080 on the end, so like http://192.168.10.5:8080 (if the local IP was 192.168.10.5)
18. Tune the model for your parameters:
    ```
    ./build/bin/llama-server -m ../llm-models/granite-3.2-8b-instruct-Q8_0.gguf -ngl 99 -nkvo -c 64000  --host :: --port 8080
    ```
    This argument string should allow the model (Granite 3.2, which has a relatively large context window in general, is getting a 64k token context window here) to hybridize its reliance on both CPU and GPU power:
      `-ngl 99` flag means "load as many layers into the GPU as possible" and `-nkvo` means "do not offload context tokens into GPU" which is important if your ngl flag is already consuming most of your VRAM. You have to make sure the MiB count for layer loads doesn't exceed the capacity of your GPU VRAM. Once you've got a model running, there's an endless collection of flags you can tweak to get it to perform up to par on your system, utilizing all the processing power and memory you want to make available.
19. Run some benchmarks and post to one of the llama.cpp benchmark discussion threads (here: https://github.com/ggml-org/llama.cpp/discussions/10879 and here: https://github.com/ggml-org/llama.cpp/discussions/4167)
    ```
    ./build/bin/llama-bench -m ../llm-models/llama2-7b-chat-q8_0.gguf -m ../llm-models/llama-2-7b-chat.Q4_0.gguf -p 512 -n 128 -ngl 99 2> /dev/null
    ```
    (requires you've got the right filenames loaded up -- these tests run on two different quantizations of Meta's llama 2 7b parameter model.) Here's how they run on my machine for reference:
    | model                          |       size |     params | backend    | threads |          test |                  t/s |
    | ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
    | llama 7B Q8_0                  |   6.67 GiB |     6.74 B | Vulkan,BLAS |      12 |         pp512 |         68.55 ± 0.25 |
    | llama 7B Q8_0                  |   6.67 GiB |     6.74 B | Vulkan,BLAS |      12 |         tg128 |         11.05 ± 0.03 |
    | llama 7B Q4_0                  |   3.56 GiB |     6.74 B | Vulkan,BLAS |      12 |         pp512 |         68.86 ± 0.16 |
    | llama 7B Q4_0                  |   3.56 GiB |     6.74 B | Vulkan,BLAS |      12 |         tg128 |         16.73 ± 0.05 |
    
    build: d3bd7193 (5092)

    And if you want to test CPU-only, change that ngl flag to value 0:
    ```
    ./build/bin/llama-bench -m ../llm-models/llama2-7b-chat-q8_0.gguf -m ../llm-models/llama-2-7b-chat.Q4_0.gguf -p 512 -n 128 -ngl 0 2> /dev/null
    ```
    Here's my CPU-only results on the 2.7Ghz 12-core Xeon for comparison:
    | model                          |       size |     params | backend    | threads |          test |                  t/s |
    | ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
    | llama 7B Q8_0                  |   6.67 GiB |     6.74 B | Vulkan,BLAS |      12 |         pp512 |         25.87 ± 0.56 |
    | llama 7B Q8_0                  |   6.67 GiB |     6.74 B | Vulkan,BLAS |      12 |         tg128 |          6.85 ± 0.00 |
    | llama 7B Q4_0                  |   3.56 GiB |     6.74 B | Vulkan,BLAS |      12 |         pp512 |         26.17 ± 0.06 |
    | llama 7B Q4_0                  |   3.56 GiB |     6.74 B | Vulkan,BLAS |      12 |         tg128 |         10.85 ± 0.01 |
    
    build: d3bd7193 (5092)
21. Finally, DO consider installing software like Macs Fan Control, TG Pro, or iStat Menus that can help you dial up your machine's fan speed to cool down those GPUs if you're going to be running them a lot. Apparently the GPUs on these old Mac Pros had a bad habit of burning out from overuse since Apple allegedly optimized silence over heat reduction. And they can definitely get dangerously hot if you put your Trash Can in heavy production for running LLMs (in other words, try to avoid creating a dumpster fire).

## Llama 4:
So you want to run the new fancy Llama 4 model. This requires some trial-and-error tuning. First step is trial-and-erroring your way to getting a set of run conditions that will allow the model to produce a non-garbage stream of tokens in response to a prompt. Then, we start working on understanding the give-and-take of various flags that will get us to longer context windows (Llama 4's context window max is like 10M tokens -- arguably its biggest differentiator against competitors). Then once we've got the context window dialed in and we have verified we know where we start to go over the the edge of instability, we can start tuning for purpose. The easiest way to do this is via llama-bench, which runs an input test (pp512) and an output test (tg128). If you are thinking about using Llama 4 with a long context window I'm assuming you may be planning on a RAG setup, in which case we want to probably maximize pp512 over tg128 if there's a tradeoff in performance to be had between input and output. All that said, nothing about Llama4 on a 12-year-old computer will be fast, so set expectations appropriately.

_Note: All tests and tweaking here will be done on unsloth's 2.71-bit quantization of Llama 4 Scout: https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF?show_file_info=Llama-4-Scout-17B-16E-Instruct-UD-Q2_K_XL.gguf_

### Step 1: getting it running without garbage output:
Once you download the GGUF, try running this:
`./build/bin/llama-server -m ../llm-models/Llama-4-Scout-17B-16E-Instruct-UD-Q2_K_XL.gguf -t 11 -ngl 11 -nkvo`
* `-t 11` runs 11 threads (though people -- and llama.cpp's authors -- say you're supposed to use the same number of threads as cores, 11 threads always produces marginally higher benchmark results than 12 threads for my 12-core CPU; experiment with this -- if you have fewer cores, start with the # of cores and play around a little to find out what's best)
* `-ngl 11` unloads 11 layers into the GPUs (it takes about 9 GB of layers into the VRAM buffer, optimized for my 12 GB of VRAM on the dual D700s; if you have D500s or D300s, start with fewer layers)
* `-nkvo` forces the CPU and RAM to take on the context challenge. This flag has been the only way I've found thus far to load a model significantly larger than the VRAM.

Hit up the provided URL (http://127.0.0.1:8080) in a browser and see whether it provides a "normal" response -- i.e. generates no garbage characters, it responds in the language in which you prompted it, uses correct grammar, and doesn't start talking to itself.

If you start to observe abnormal or erratic behavior, adjust values for `-t` and `-ngl` until you start to see some success. using `-ngl 0` will put all the work on the CPUs, which should be slow but productive. `-ngl 1` keeps the GPUs in play but minimally so. Find your functional limit by incrementing up from `-ngl 1` until the model starts spitting garbage, and then backing off by one layer -- that's your upper bound.

Here's a benchmark for this run, using code: `./build/bin/llama-bench -m ../llm-models/Llama-4-Scout-17B-16E-Instruct-UD-Q2_K_XL.gguf -t 11 -ngl 11 -nkvo 1`
| model                          |       size |     params | backend    | threads |          nkvo |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | ------------: | -------------------: |
| llama4 17Bx16E (Scout) Q2_K - Medium |  39.65 GiB |   107.77 B | Vulkan,BLAS |      11 |             1 |         pp512 |          9.43 ± 0.13 |
| llama4 17Bx16E (Scout) Q2_K - Medium |  39.65 GiB |   107.77 B | Vulkan,BLAS |      11 |             1 |         tg128 |          4.45 ± 0.01 |

With `-ngl 1` instead of `-ngl 11` but holding everything else steady, we can see both benchmark rates decline:
| model                          |       size |     params | backend    | threads |          nkvo |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | ------------: | -------------------: |
| llama4 17Bx16E (Scout) Q2_K - Medium |  39.65 GiB |   107.77 B | Vulkan,BLAS |      11 |             1 |         pp512 |          8.80 ± 0.14 |
| llama4 17Bx16E (Scout) Q2_K - Medium |  39.65 GiB |   107.77 B | Vulkan,BLAS |      11 |             1 |         tg128 |          4.22 ± 0.01 |

And here's `-ngl 0` where the CPUs do all the work:
| model                          |       size |     params | backend    | threads |          nkvo |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | ------------: | -------------------: |
| llama4 17Bx16E (Scout) Q2_K - Medium |  39.65 GiB |   107.77 B | Vulkan,BLAS |      11 |             1 |         pp512 |          8.78 ± 0.14 |
| llama4 17Bx16E (Scout) Q2_K - Medium |  39.65 GiB |   107.77 B | Vulkan,BLAS |      11 |             1 |         tg128 |          4.12 ± 0.00 |

The main thing we learn here is that the GPUs are not carrying very much of the load of token generation on this 17B-parameter model. In terms of t/s, between `-ngl 0` and `-ngl 13` (benched below) we see a ~9% increase from 8.78 ± 0.14 to 9.59 ± 0.11 on parsing and between `-ngl 0` and `-ngl 11` an ~8% rise from 4.12 ± 0.00 to 4.45 ± 0.01 in token generation. It's possible we have to go back to the compiler or graphics driver to optimize here but for the time being let's work with what we've got.

I noticed both `-ngl 1` and `-ngl 11` tend to lean more heavily on just one of the GPUs, so we'll make a note to try to share the love between both. GPU device #0 is the default primary, which is the GPU that does most of the graphics work on the computer (system settings and activity monitor show it as the monitor-attached device responsible for rendering video). We want to nudge it to lean more heavily on the second GPU (device 1). Using `--tensor-split 4/5` should put 5/9ths (5/(4+5)) of the layer load on GPU #1 vs GPU #0. It shouldn't significantly affect performance but it will maybe preserve the life of GPU #0. We could also use the `--main-gpu 1` argument to ask it to primarily task GPU #1.
| model                          |       size |     params | backend    | threads |          nkvo | ts           |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | ------------ | ------------: | -------------------: |
| llama4 17Bx16E (Scout) Q2_K - Medium |  39.65 GiB |   107.77 B | Vulkan,BLAS |      11 |             1 | 4.00/5.00    |         pp512 |          9.46 ± 0.12 |
| llama4 17Bx16E (Scout) Q2_K - Medium |  39.65 GiB |   107.77 B | Vulkan,BLAS |      11 |             1 | 4.00/5.00    |         tg128 |          4.28 ± 0.01 |

Looking at my llama-server log using these flags I see this under load_tensors:
```
load_tensors: offloading 11 repeating layers to GPU
load_tensors: offloaded 11/49 layers to GPU
load_tensors:   CPU_Mapped model buffer size = 31730.07 MiB
load_tensors:      Vulkan1 model buffer size =  4034.57 MiB
load_tensors:      Vulkan0 model buffer size =  4841.48 MiB
```
I notice that `-ngl 11` is offloading about 8.9 GB to the GPUs -- since I have 12 GB of space I can theoretically go up a couple notches. `-ngl 14` gets me just under the 12GB threshold with the following:
```
load_tensors: offloading 14 repeating layers to GPU
load_tensors: offloaded 14/49 layers to GPU
load_tensors:   CPU_Mapped model buffer size = 29309.33 MiB
load_tensors:      Vulkan0 model buffer size =  5648.40 MiB
load_tensors:      Vulkan1 model buffer size =  5648.40 MiB
```
Just for kicks, let's see what that does on a benchmark run:
| model                          |       size |     params | backend    | threads |          nkvo | ts           |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | ------------ | ------------: | -------------------: |
| llama4 17Bx16E (Scout) Q2_K - Medium |  39.65 GiB |   107.77 B | Vulkan,BLAS |      11 |             1 | 4.00/5.00    |         pp512 |          9.42 ± 0.13 |
| llama4 17Bx16E (Scout) Q2_K - Medium |  39.65 GiB |   107.77 B | Vulkan,BLAS |      11 |             1 | 4.00/5.00    |         tg128 |          0.84 ± 0.00 |

Well that is absolutely terrible. The prompt parsing went about as per normal with more layers; the token generation task dropped to a slow crawl. We'll drop it back to `-ngl 13` and see if things improve:
| model                          |       size |     params | backend    | threads |          nkvo | ts           |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | ------------ | ------------: | -------------------: |
`-ngl 13`:
| llama4 17Bx16E (Scout) Q2_K - Medium |  39.65 GiB |   107.77 B | Vulkan,BLAS |      11 |             1 | 4.00/5.00    |         pp512 |          9.59 ± 0.11 |
| llama4 17Bx16E (Scout) Q2_K - Medium |  39.65 GiB |   107.77 B | Vulkan,BLAS |      11 |             1 | 4.00/5.00    |         tg128 |          4.15 ± 0.01 |
`-ngl 12`:
| llama4 17Bx16E (Scout) Q2_K - Medium |  39.65 GiB |   107.77 B | Vulkan,BLAS |      11 |             1 | 4.00/5.00    |         pp512 |          9.54 ± 0.15 |
| llama4 17Bx16E (Scout) Q2_K - Medium |  39.65 GiB |   107.77 B | Vulkan,BLAS |      11 |             1 | 4.00/5.00    |         tg128 |          4.13 ± 0.00 |

Hmm. These two appear to be performing near-identically, despite having different layer loads. Let's look into the CLI load log and see what's happening under these conditions:
```
load_tensors: offloading 13 repeating layers to GPU
load_tensors: offloaded 13/49 layers to GPU
load_tensors:   CPU_Mapped model buffer size = 30116.24 MiB
load_tensors:      Vulkan0 model buffer size =  4841.48 MiB
load_tensors:      Vulkan1 model buffer size =  5648.40 MiB
...
load_tensors: offloading 12 repeating layers to GPU
load_tensors: offloaded 12/49 layers to GPU
load_tensors:   CPU_Mapped model buffer size = 30923.16 MiB
load_tensors:      Vulkan0 model buffer size =  4841.48 MiB
load_tensors:      Vulkan1 model buffer size =  4841.48 MiB
```
`-ngl 12` consumes about 0.8GB less VRAM but doesn't meaningfully impact production time compared to `-ngl 13`. If we wanted to create more VRAM space for some reason, we could drop down to `-ngl 1` and just keep a single layer in a GPU.


### Step 2: getting it running with a larger context window:
By default it's using a 4096 token context window. So you're getting the impressive training of the 17B-parameter model but not taking advantage of the larger context (upper context limit is 10,485,760 tokens). We want to drop `-ngl` to 11 again to create more VRAM space. We set context with `-c X` or `--ctx-size X`. Let's see what happens with a 64,000-token window to start:

`./build/bin/llama-server -m ../llm-models/Llama-4-Scout-17B-16E-Instruct-UD-Q2_K_XL.gguf -t 11 -ngl 11 -nkvo --ctx-size 64000`

(in http://127.0.0.1:8080):
```
Could you give me a Shakespearean sonnet about aircraft carriers?

✍️ Edit
📋 Copy
A Shakespearean sonnet, with a traditional rhyme scheme of ABAB CDCD EFEF GG:

When mighty aircraft carriers sail the sea,
Their decks a-throng with planes that eagerly
Await the signal to take to the air,
And with a roar, their engines loudly share

Their power and strength, a wonder to behold,
A floating fortress, impenetrable and cold,
Their guns and missiles, a defense to hold
Against the foe, who dare their power to unfold

But oh, how fragile life upon them lies,
A single hit, a catastrophic demise,
Their steel hulls, a shield against the tide,
But vulnerable to fire, that most dire

Yet still they sail, a symbol of our might,
And keep the seas, in freedom's golden light.

How's that?

🔄 Regenerate
📋 Copy
```
Not exactly right but pretty decent first attempt. This prompt exchange doesn't show the full context being exercised but demonstrates it's at least thinking correctly.

Ramping up the context window to 256000 tokens with: `./build/bin/llama-server -m ../llm-models/Llama-4-Scout-17B-16E-Instruct-UD-Q2_K_XL.gguf -t 11 -ngl 11 -nkvo --ctx-size 256000` -- notably the launcher protests we're not taking full advantage of the model's capacity with this: `llama_context: n_ctx_per_seq (256000) < n_ctx_train (10485760) -- the full capacity of the model will not be utilized`

With a 256k context window it's sharing some additional information:
```
llama_model_load_from_file_impl: using device Vulkan0 (AMD Radeon HD - FirePro D700) - 6144 MiB free
llama_model_load_from_file_impl: using device Vulkan1 (AMD Radeon HD - FirePro D700) - 6144 MiB free
...
load_tensors: offloading 11 repeating layers to GPU
load_tensors: offloaded 11/49 layers to GPU
load_tensors:   CPU_Mapped model buffer size = 31730.07 MiB
load_tensors:      Vulkan1 model buffer size =  4034.57 MiB
load_tensors:      Vulkan0 model buffer size =  4841.48 MiB
...
llama_context: n_ctx_per_seq (256000) < n_ctx_train (10485760) -- the full capacity of the model will not be utilized
llama_context:        CPU  output buffer size =     0.77 MiB
init: kv_size = 256000, offload = 0, type_k = 'f16', type_v = 'f16', n_layer = 48, can_shift = 1
init:        CPU KV buffer size = 48000.00 MiB
llama_context: KV self size  = 48000.00 MiB, K (f16): 24000.00 MiB, V (f16): 24000.00 MiB
ggml_vulkan: Failed to allocate pinned memory.
ggml_vulkan: Requested buffer size exceeds device memory allocation limit: ErrorOutOfDeviceMemory
llama_context:    Vulkan0 compute buffer size =   616.06 MiB
llama_context:    Vulkan1 compute buffer size =    72.00 MiB
llama_context:        CPU compute buffer size =   404.62 MiB
llama_context: Vulkan_Host compute buffer size = 21014.01 MiB
llama_context: graph nodes  = 2514
llama_context: graph splits = 749 (with bs=512), 26 (with bs=1)
common_init_from_params: setting dry_penalty_last_n to ctx_size = 256000
```
Compare the bottom section for `-c 256000` to that for `c- 64000`:
```
llama_context: n_ctx_per_seq (64000) < n_ctx_train (10485760) -- the full capacity of the model will not be utilized
llama_context:        CPU  output buffer size =     0.77 MiB
init: kv_size = 64000, offload = 0, type_k = 'f16', type_v = 'f16', n_layer = 48, can_shift = 1
init:        CPU KV buffer size = 12000.00 MiB
llama_context: KV self size  = 12000.00 MiB, K (f16): 6000.00 MiB, V (f16): 6000.00 MiB
ggml_vulkan: Failed to allocate pinned memory.
ggml_vulkan: Requested buffer size exceeds device memory allocation limit: ErrorOutOfDeviceMemory
llama_context:    Vulkan0 compute buffer size =   616.06 MiB
llama_context:    Vulkan1 compute buffer size =    72.00 MiB
llama_context:        CPU compute buffer size =   404.62 MiB
llama_context: Vulkan_Host compute buffer size =  5264.01 MiB
llama_context: graph nodes  = 2514
llama_context: graph splits = 749 (with bs=512), 26 (with bs=1)
common_init_from_params: setting dry_penalty_last_n to ctx_size = 64000
```
I think this means it's going to use disk swap for context until I drop the context size to the point the KV buffer request size fits in the sum of the available compute buffers.

### Llama 4 OpenAI Clone server, slow and steady:
To run a Llama 4 server accessible to your local network on port 80 (i.e., responding to OpenAI API calls from programs like Cursor for which you can't set a port) and hold VRAM space and GPU bandwidth to run other models in parallel using the GPU, try the following:
```
./build/bin/llama-server -m ../llm-models/Llama-4-Scout-17B-16E-Instruct-UD-Q2_K_XL.gguf \
    -nkvo \
    --threads 10 \
    --ctx-size 65536 \
    --n-gpu-layers 0 \
    -ot "([0-9][0-9]).ffn_.*_exps.=CPU" \
    --seed 3407 \
    --prio 3 \
    --temp 0.6 \
    --min-p 0.01 \
    --top-p 0.9 \
    --host :: \
    --port 80
```
(ref: Unsloth's Guide to Run and Fine-Tune Llama 4 [https://docs.unsloth.ai/basics/tutorial-how-to-run-and-fine-tune-llama-4])
