import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RelationAwareTripletModel(nn.Module):
    def __init__(self, base_encoder: nn.Module, relations: list[str], projection_dim: int, relation_dim: int):
        super().__init__()
        self.base_encoder = base_encoder
        self.relations = relations
        self.relation_to_id = {relation: idx for idx, relation in enumerate(relations)}
        self.projection_head = ProjectionHead(base_encoder.output_dim, projection_dim)
        self.relation_embeddings = nn.Embedding(len(relations), relation_dim)
        self.relation_adapter = nn.Linear(projection_dim + relation_dim, projection_dim)

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        return self.projection_head(self.base_encoder.encode_texts(texts))

    def relation_ids(self, relations: list[str], device: torch.device) -> torch.Tensor:
        ids = [self.relation_to_id[relation] for relation in relations]
        return torch.tensor(ids, dtype=torch.long, device=device)

    def compose_anchor(self, anchor_projection: torch.Tensor, relation_ids: torch.Tensor) -> torch.Tensor:
        relation_vectors = self.relation_embeddings(relation_ids)
        combined = torch.cat([anchor_projection, relation_vectors], dim=-1)
        return self.relation_adapter(combined)
