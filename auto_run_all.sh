#partial
python run.py --model gat --dataset PascalVOC-SP --metric macrof1 --criterion weighted_cross_entropy --partial 1 --k 5 --dropout 0.2 --space 2
python run.py --model gat --dataset peptides-func --task graph --metric ap --partial 1 --k 5 --dropout 0.2 --space 2
python run.py --model gat --datatype 3d --dataset psb --task graph --partial 1 --k 5 --dropout 0.2 --space 3 --bz 16 --metric ap --add_edges 5

python run.py --model gcn --datatype 3d --dataset psb --task graph --metric ap --add_edges 5 --bz 16
python run.py --model gatedgcn --datatype 3d --dataset psb --task graph --metric ap --add_edges 5 --bz 16
python run.py --model gin --datatype 3d --dataset psb --task graph --metric ap --add_edges 5 --bz 16
python run.py --model gat --datatype 3d --dataset psb --task graph --metric ap --add_edges 5 --bz 16
python run.py --model san --datatype 3d --dataset psb --task graph --metric ap --add_edges 5 --bz 16

#5 for k nearest neighbor edges added to each node
python run.py --model gcn --datatype 3d --dataset psb --task graph --metric ap --add_edges 5 --bz 8
python run.py --model gatedgcn --datatype 3d --dataset psb --task graph --metric ap --add_edges 5 --bz 8
python run.py --model gin --datatype 3d --dataset psb --task graph --metric ap --add_edges 5 --bz 8
python run.py --model gat --datatype 3d --dataset psb --task graph --metric ap --add_edges 5 --bz 8
python run.py --model san --datatype 3d --dataset psb --task graph --metric ap --add_edges 5 --bz 8

#Encoding models
python run.py --model gcn --dataset PascalVOC-SP --metric macrof1 --criterion weighted_cross_entropy --encode lap --encode_k 10
python run.py --model gatedgcn --dataset PascalVOC-SP --metric macrof1 --criterion weighted_cross_entropy --encode lap --encode_k 10
python run.py --model gin --dataset PascalVOC-SP --metric macrof1 --criterion weighted_cross_entropy --encode lap --encode_k 10
python run.py --model san --dataset PascalVOC-SP --metric macrof1 --criterion weighted_cross_entropy --encode lap --encode_k 10

python run.py --model gcn --dataset peptides-func --task graph --metric ap --encode lap --encode_k 10
python run.py --model gatedgcn --dataset peptides-func --task graph --metric ap --encode lap --encode_k 10
python run.py --model gin --dataset peptides-func --task graph --metric ap --encode lap --encode_k 10
python run.py --model san --dataset peptides-func --task graph --metric ap --encode lap --encode_k 10

#RWSE models
python run.py --model gcn --dataset PascalVOC-SP --metric macrof1 --criterion weighted_cross_entropy --encode walk --encode_k 10
python run.py --model gatedgcn --dataset PascalVOC-SP --metric macrof1 --criterion weighted_cross_entropy --encode walk --encode_k 10
python run.py --model gin --dataset PascalVOC-SP --metric macrof1 --criterion weighted_cross_entropy --encode walk --encode_k 10
python run.py --model gat --dataset PascalVOC-SP --metric macrof1 --criterion weighted_cross_entropy --encode walk --encode_k 10
python run.py --model san --dataset PascalVOC-SP --metric macrof1 --criterion weighted_cross_entropy --encode walk --encode_k 10

python run.py --model gcn --dataset peptides-func --task graph --metric ap --encode walk --encode_k 10
python run.py --model gatedgcn --dataset peptides-func --task graph --metric ap --encode walk --encode_k 10
python run.py --model gin --dataset peptides-func --task graph --metric ap --encode walk --encode_k 10
python run.py --model gat --dataset peptides-func --task graph --metric ap --encode walk --encode_k 10
python run.py --model san --dataset peptides-func --task graph --metric ap --encode walk --encode_k 10

python run.py --model gcn --datatype 3d --dataset psb --task graph --metric ap --add_edges 5 --bz 16 --encode walk --encode_k 10
python run.py --model gatedgcn --datatype 3d --dataset psb --task graph --metric ap --add_edges 5 --bz 16 --encode walk --encode_k 10
python run.py --model gin --datatype 3d --dataset psb --task graph --metric ap --add_edges 5 --bz 16 --encode walk --encode_k 10
python run.py --model gat --datatype 3d --dataset psb --task graph --metric ap --add_edges 5 --bz 16 --encode walk --encode_k 10
python run.py --model san --datatype 3d --dataset psb --task graph --metric ap --add_edges 5 --bz 16 --encode walk --encode_k 10

#edge+lap models
python run.py --model gcn --dataset PascalVOC-SP --metric macrof1 --criterion weighted_cross_entropy --encode lap --encode_k 10 --add_edges 1
python run.py --model gatedgcn --dataset PascalVOC-SP --metric macrof1 --criterion weighted_cross_entropy --encode lap --encode_k 10 --add_edges 1
python run.py --model gin --dataset PascalVOC-SP --metric macrof1 --criterion weighted_cross_entropy --encode lap --encode_k 10 --add_edges 1
python run.py --model san --dataset PascalVOC-SP --metric macrof1 --criterion weighted_cross_entropy --encode lap --encode_k 10 --add_edges 1

python run.py --model gcn --dataset peptides-func --task graph --metric ap --encode lap --encode_k 10 --add_edges 1
python run.py --model gatedgcn --dataset peptides-func --task graph --metric ap --encode lap --encode_k 10 --add_edges 1
python run.py --model gin --dataset peptides-func --task graph --metric ap --encode lap --encode_k 10 --add_edges 1
python run.py --model san --dataset peptides-func --task graph --metric ap --encode lap --encode_k 10 --add_edges 1

#edge+rwse models
python run.py --model gcn --dataset PascalVOC-SP --metric macrof1 --criterion weighted_cross_entropy --encode walk --encode_k 10 --add_edges 1
python run.py --model gatedgcn --dataset PascalVOC-SP --metric macrof1 --criterion weighted_cross_entropy --encode walk --encode_k 10 --add_edges 1
python run.py --model gin --dataset PascalVOC-SP --metric macrof1 --criterion weighted_cross_entropy --encode walk --encode_k 10 --add_edges 1
python run.py --model gat --dataset PascalVOC-SP --metric macrof1 --criterion weighted_cross_entropy --encode walk --encode_k 10 --add_edges 1
python run.py --model san --dataset PascalVOC-SP --metric macrof1 --criterion weighted_cross_entropy --encode walk --encode_k 10 --add_edges 1

python run.py --model gcn --dataset peptides-func --task graph --metric ap --encode walk --encode_k 10 --add_edges 1
python run.py --model gatedgcn --dataset peptides-func --task graph --metric ap --encode walk --encode_k 10 --add_edges 1
python run.py --model gin --dataset peptides-func --task graph --metric ap --encode walk --encode_k 10 --add_edges 1
python run.py --model gat --dataset peptides-func --task graph --metric ap --encode walk --encode_k 10 --add_edges 1
python run.py --model san --dataset peptides-func --task graph --metric ap --encode walk --encode_k 10 --add_edges 1

python run.py --model gcn --datatype 3d --dataset psb --task graph --metric ap --add_edges 5 --bz 8 --encode walk --encode_k 10
python run.py --model gatedgcn --datatype 3d --dataset psb --task graph --metric ap --add_edges 5 --bz 8 --encode walk --encode_k 10
python run.py --model gin --datatype 3d --dataset psb --task graph --metric ap --add_edges 5 --bz 8 --encode walk --encode_k 10
python run.py --model gat --datatype 3d --dataset psb --task graph --metric ap --add_edges 5 --bz 8 --encode walk --encode_k 10
python run.py --model san --datatype 3d --dataset psb --task graph --metric ap --add_edges 5 --bz 8 --encode walk --encode_k 10

