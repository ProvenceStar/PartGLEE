# Tutorial for Training

PartGLEE series are trained based on GLEE weights, before you start training, please first download the GLEE weights under the "checkpoint" folder, then use our converter to transform GLEE weights into PartGLEE pretrain weights.

|       Name        |   Backbone   |                            Weight                            |
| :---------------: | :----------: | :----------------------------------------------------------: |
| GLEE-Lite-scaleup |  ResNet-50   | [Model](https://huggingface.co/spaces/Junfeng5/GLEE_demo/resolve/main/MODEL_ZOO/GLEE_Lite_scaleup.pth) |
| GLEE-Plus-scaleup |    Swin-L    | [Model](https://huggingface.co/spaces/Junfeng5/GLEE_demo/resolve/main/MODEL_ZOO/GLEE_Plus_scaleup.pth) |

Then run the command below to convert the pretrained weight:
```bash
python3 projects/PartGLEE/tools/converter.py --glee_weight_path <glee-weight-path> --output_path <converted-weight-path>
```
Replace `<glee-weight-path>` and `<converted-weight-path>`. For more detail please check `converter.py`.

## Single machine training:

For training on a single machine, you can execute the following command:

```bash
python3 projects/PartGLEE/train_net.py --config-file projects/PartGLEE/configs/Training/<your-config.yaml> --num-gpus 8
```

Replace `<your-config.yaml>` with your actual configuration file.

## Multiple machines training:

Our standard setup involves training on multiple machines (32 x A100), for which you can use the distributed training script:

```bash
python3 launch.py --nn <num_machines> --port <PORT> --worker_rank <Global_Rank> --master_address $<MASTER_ADDRESS>  --config-file projects/PartGLEE/configs/Training/<your-config.yaml>
```

Here, `<num_machines>` should be replaced with the number of machines you intend to use, `<MASTER_ADDRESS>` should be the IP address of node 0. `<PORT>` should be the same among multiple nodes. , and `<your-config.yaml>` with your actual configuration file.