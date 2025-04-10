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
