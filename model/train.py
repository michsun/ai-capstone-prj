from model import VRNet, VRDataLoader


def data_preprocessing():
    return

def train(epochs):
    model = VRNet()
    model.train()
    
    # TODO: Add train details here
    parameter = "Something"

if __name__ == "__main__":
    data_dir = "../data/simulated-examples"
    rgbs, actions = VRDataLoader(data_dir=data_dir).load_data()
    print(actions)
    # train()