# COMP34120 AI and Games Coursework

We will be running each within this specific Dockerfile image and with this specific Docker environment. So make sure
your agent will work within this container environment and under the constrain.

Inside the Dockerfile, two versions of PyTorch and Tensorflow are available. Select the one you need based on the availability and use of the GPU.

To build Dockerfile.

```bash
docker build --build-arg UID=$UID -t hex .
```

The building process will take a while.

To run the container use:

```bash
docker run --cpus=8 --memory=8G -v "$(pwd)":/home/hex --name hex --rm -it hex /bin/bash
```

If you need GPU access, pass `--runtime=nvidia` to the docker run command.

The current repo will be mapped to `/home/hex` within the container.
If you `cd hex` you should be able to see all your
local file. Any changes made to that directory will reflect to your system directory. This will be the command we use
to create the running environment for playing each game, so your agent can at most you 8 CPUs and uses 8 GB of memory.

To run a game of Hex, you can use:

```bash
python3 Hex.py
```

By default, two `agents/DefaultAgents/NaiveAgent.py` agent will play against each. To see all the available options and
help message use `python3 Hex.py --help`.

To exit the docker container you can simply do `exit`. This will stop the container.

To enter the container again you can simply use:

```bash
docker start -i hex
```

To run the test suite, you can use:

```bash
python3 -m unittest discover
```

To use a GPU, use **CUDA 12.3.0** with the preinstalled **TensorFlow 2.19.0** and **PyTorch 2.5.1+cu121**, or ensure any other versions you install are compatible with them.

## Misc

PDF doc link: [typst](https://typst.app/project/wimHW-RlEYIYkqEVJWgrXC)
