#Laplacian encoding and Edge Addition Duo
python run.py --model gcn --dataset PascalVOC-SP --metric macrof1 --criterion weighted_cross_entropy --add_edges 1 --encode lap --encode_k=10
python run.py --model gin --dataset PascalVOC-SP --metric macrof1 --criterion weighted_cross_entropy --add_edges 1 --encode lap --encode_k=10
python run.py --model gat --dataset PascalVOC-SP --metric macrof1 --criterion weighted_cross_entropy --add_edges 1 --encode lap --encode_k=10
python run.py --model san --dataset PascalVOC-SP --metric macrof1 --criterion weighted_cross_entropy --add_edges 1 --encode lap --encode_k=10

python run.py --model gcn --dataset peptides-func --task graph --metric ap --add_edges 1 --encode lap --encode_k=10
python run.py --model gin --dataset peptides-func --task graph --metric ap --add_edges 1 --encode lap --encode_k=10
python run.py --model gat --dataset peptides-func --task graph --metric ap --add_edges 1 --encode lap --encode_k=10
python run.py --model san --dataset peptides-func --task graph --metric ap --add_edges 1 --encode lap --encode_k=10