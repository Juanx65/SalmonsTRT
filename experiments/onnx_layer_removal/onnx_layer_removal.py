import onnx
from onnx import helper

# Cargar el modelo ONNX
model_path = 'weights/best.onnx'
model = onnx.load(model_path)

# Explorar el grafo para identificar las capas/nodos a eliminar
""" print("Nodos actuales en el modelo:")
for node in model.graph.node:
    print(node.name, node.op_type) """

# Aquí deberás identificar los nodos por nombre o tipo y eliminarlos.
# Este es un ejemplo de cómo podrías eliminar un nodo por nombre (ajusta los nombres según corresponda):
nodes_to_remove = ['/model.22/proto/cv3/act/Sigmoid', '/model.22/proto/cv3/act/Mul']  # Reemplaza esto con los nombres reales de tus nodos '/model.22/proto/cv3/conv/Conv', 
nodes = [node for node in model.graph.node if node.name not in nodes_to_remove]

# Crear un nuevo grafo sin los nodos eliminados
new_graph = helper.make_graph(nodes, model.graph.name, model.graph.input, model.graph.output, model.graph.initializer)

""" print("Nodos actuales en el modelo:")
for node in new_graph.node:
    print(node.name, node.op_type) """

# Crear un nuevo modelo ONNX con el grafo modificado
new_model = helper.make_model(new_graph)
model = new_model

nueva_salida = '/model.22/proto/cv3/conv/Conv'

# Buscar la capa por nombre y obtener detalles para la nueva salida
for node in model.graph.node:
    if node.name == nueva_salida:
        node.output[0] = 'output1'
        break


# Verificar el modelo
onnx.checker.check_model(model)

# Guardar el modelo modificado
onnx.save(model, 'weights/best_new.onnx')
