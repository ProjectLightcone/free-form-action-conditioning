export interface DiagramNode {
	id: string;
	label: string[];
	kind?: 'input' | 'encoder' | 'latent' | 'operator' | 'output' | 'target' | 'loss' | 'warm';
	width?: number;
	height?: number;
}

export interface DiagramEdge {
	from: string;
	to: string;
}

export interface DiagramDefinition {
	title: string;
	subtitle?: string;
	caption: string;
	rankdir?: 'LR' | 'TB';
	nodesep?: number;
	ranksep?: number;
	nodes: DiagramNode[];
	edges: DiagramEdge[];
}

export const trainingLoopDiagram: DiagramDefinition = {
	title: 'TEPA Training Loop',
	subtitle: 'Predict a conditioned target embedding from context plus condition.',
	caption:
		'Figure 1. TEPA factorizes context, condition, and target into separate embedded terms. The target path is a parallel training signal, not an inference-time decoder.',
	rankdir: 'TB',
	nodesep: 36,
	ranksep: 54,
	nodes: [
		{
			id: 'context',
			label: ['Context x', 'image, text,', 'sensor, table'],
			kind: 'input',
			width: 170,
			height: 86,
		},
		{ id: 'eContext', label: ['Context encoder', 'E_context'], kind: 'encoder', width: 170 },
		{ id: 'zContext', label: ['z_context'], kind: 'latent', width: 145, height: 68 },
		{
			id: 'condition',
			label: ['Condition c', 'instruction, action,', 'query, horizon'],
			kind: 'input',
			width: 170,
			height: 86,
		},
		{ id: 'eCondition', label: ['Condition encoder', 'E_condition'], kind: 'encoder', width: 170 },
		{ id: 'zCondition', label: ['z_condition'], kind: 'latent', width: 145, height: 68 },
		{
			id: 'predictor',
			label: ['Predictor P', 'latent transition /', 'task operator'],
			kind: 'operator',
			width: 210,
			height: 96,
		},
		{
			id: 'zHat',
			label: ['z_hat_target', 'predicted conditioned', 'outcome'],
			kind: 'output',
			width: 190,
			height: 86,
		},
		{
			id: 'target',
			label: ['Conditioned target y_c', 'answer + uncertainty', '+ support factors'],
			kind: 'target',
			width: 210,
			height: 90,
		},
		{ id: 'eTarget', label: ['Target encoder', 'E_target'], kind: 'encoder', width: 170 },
		{ id: 'zTarget', label: ['z_target'], kind: 'latent', width: 145, height: 68 },
		{
			id: 'loss',
			label: ['Loss', 'd(z_hat_target, z_target)', '+ lambda R'],
			kind: 'loss',
			width: 230,
			height: 88,
		},
	],
	edges: [
		{ from: 'context', to: 'eContext' },
		{ from: 'eContext', to: 'zContext' },
		{ from: 'zContext', to: 'predictor' },
		{ from: 'condition', to: 'eCondition' },
		{ from: 'eCondition', to: 'zCondition' },
		{ from: 'zCondition', to: 'predictor' },
		{ from: 'predictor', to: 'zHat' },
		{ from: 'zHat', to: 'loss' },
		{ from: 'target', to: 'eTarget' },
		{ from: 'eTarget', to: 'zTarget' },
		{ from: 'zTarget', to: 'loss' },
	],
};

export const mllmComparisonDiagram: DiagramDefinition = {
	title: 'Why Not Just a Multimodal LLM?',
	subtitle: 'The systems are complementary: one is a universal interface; the other is a latent consequence predictor.',
	caption:
		'Figure 2. Multimodal LLMs are natural interfaces; TEPA is proposed as a latent consequence engine.',
	rankdir: 'TB',
	nodesep: 48,
	ranksep: 52,
	nodes: [
		{ id: 'mContext', label: ['Multimodal context'], kind: 'input', width: 190 },
		{ id: 'tokens', label: ['Autoregressive', 'token loop'], kind: 'warm', width: 200 },
		{ id: 'tokenOut', label: ['Text / token output'], kind: 'warm', width: 190 },
		{ id: 'tools', label: ['Explain or', 'call tools'], kind: 'target', width: 170 },
		{ id: 'tcContext', label: ['Context + condition'], kind: 'input', width: 190 },
		{ id: 'latentPass', label: ['Latent prediction', 'pass'], kind: 'operator', width: 200 },
		{ id: 'bundle', label: ['Predicted outcome', 'bundle'], kind: 'output', width: 200 },
		{ id: 'heads', label: ['Probe, plan,', 'retrieve, decode'], kind: 'target', width: 190 },
	],
	edges: [
		{ from: 'mContext', to: 'tokens' },
		{ from: 'tokens', to: 'tokenOut' },
		{ from: 'tokenOut', to: 'tools' },
		{ from: 'tcContext', to: 'latentPass' },
		{ from: 'latentPass', to: 'bundle' },
		{ from: 'bundle', to: 'heads' },
	],
};

export const applicationSubstrateDiagram: DiagramDefinition = {
	title: 'Predicted Latents as an Application Substrate',
	subtitle: 'TEPA does not have to decode pixels. It can feed specialized heads.',
	caption:
		"Figure 3. TEPA's native output can feed decoders, probes, retrieval systems, planners, and controllers.",
	rankdir: 'LR',
	nodesep: 42,
	ranksep: 88,
	nodes: [
		{
			id: 'latent',
			label: ['z_hat_target', 'predicted conditioned', 'outcome bundle'],
			kind: 'operator',
			width: 240,
			height: 112,
		},
		{ id: 'decoder', label: ['Decoder', 'image, video,', 'waveform, state'], kind: 'target', width: 230 },
		{ id: 'probe', label: ['Probe', 'risk, budget slack,', 'variables'], kind: 'target', width: 230 },
		{ id: 'retriever', label: ['Retriever', 'nearest outcomes,', 'examples, plans'], kind: 'target', width: 230 },
		{ id: 'planner', label: ['Planner', 'score candidate', 'actions'], kind: 'target', width: 230 },
		{ id: 'controller', label: ['Controller', 'policy/value', 'head'], kind: 'target', width: 230 },
	],
	edges: [
		{ from: 'latent', to: 'decoder' },
		{ from: 'latent', to: 'probe' },
		{ from: 'latent', to: 'retriever' },
		{ from: 'latent', to: 'planner' },
		{ from: 'latent', to: 'controller' },
	],
};

export const actionConditioningNoteDiagram: DiagramDefinition = {
	title: 'Action-Conditioned Latent Prediction',
	subtitle:
		'The experiment compares explicit condition factorization against a context-stuffed JEPA baseline.',
	caption:
		'Figure 1. The central question is whether an explicit condition path can preserve JEPA-style predictive structure while making actions, queries, and counterfactuals reusable across many context evaluations.',
	rankdir: 'TB',
	nodesep: 38,
	ranksep: 54,
	nodes: [
		{
			id: 'scene',
			label: ['Context', 'rendered puck-world', 'state'],
			kind: 'input',
			width: 190,
			height: 86,
		},
		{
			id: 'condition',
			label: ['Condition', 'action, horizon,', 'query surface'],
			kind: 'input',
			width: 190,
			height: 86,
		},
		{
			id: 'contextPath',
			label: ['Context encoder', 'CNN / ViT patches'],
			kind: 'encoder',
			width: 190,
		},
		{
			id: 'conditionPath',
			label: ['Condition encoder', 'structured or text', 'surface'],
			kind: 'encoder',
			width: 190,
			height: 84,
		},
		{
			id: 'tepa',
			label: ['TEPA predictor', 'factorized readout', 'or cross-attention'],
			kind: 'operator',
			width: 230,
			height: 96,
		},
		{
			id: 'jepa',
			label: ['Context-stuffed JEPA', 'joint context +', 'condition encoder'],
			kind: 'warm',
			width: 230,
			height: 96,
		},
		{
			id: 'target',
			label: ['Target latent', 'future outcome', 'embedding'],
			kind: 'latent',
			width: 190,
			height: 84,
		},
		{
			id: 'evals',
			label: ['Evaluation', 'endpoint metrics,', 'counterfactuals, probes'],
			kind: 'output',
			width: 230,
			height: 92,
		},
	],
	edges: [
		{ from: 'scene', to: 'contextPath' },
		{ from: 'condition', to: 'conditionPath' },
		{ from: 'contextPath', to: 'tepa' },
		{ from: 'conditionPath', to: 'tepa' },
		{ from: 'scene', to: 'jepa' },
		{ from: 'condition', to: 'jepa' },
		{ from: 'tepa', to: 'target' },
		{ from: 'jepa', to: 'target' },
		{ from: 'target', to: 'evals' },
	],
};
