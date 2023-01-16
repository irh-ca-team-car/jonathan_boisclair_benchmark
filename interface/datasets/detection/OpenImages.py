from .. import Sample
from ...detectors.Detection import Detection, Box2d
import fiftyone as fo
import fiftyone.zoo as foz


class OpenImagesDetection:
    A1Classes = None

    def classesList(self):
        if OpenImagesDetection.A1Classes is None:
            OpenImagesDetection.A1Classes =["void", *foz.load_zoo_dataset(
        "open-images-v6", split="validation", max_samples=0).get_classes("detection")]
            OpenImagesDetection.A1Classes = [f for f in OpenImagesDetection.A1Classes if not f in [
                'Accordion', 'Adhesive tape', 'Aircraft', 'Airplane', 'Alarm clock', 'Alpaca', 
                'Animal', 'Ant', 'Antelope', 'Apple', 'Armadillo', 'Artichoke', 'Auto part',
                'Axe', 'Backpack', 'Bagel', 'Baked goods', 'Balance beam', 'Ball', 'Balloon', 
                'Banana', 'Band-aid', 'Banjo', 'Barge', 'Barrel', 'Baseball bat', 'Baseball glove', 
                'Bat (Animal)', 'Bathroom accessory', 'Bathroom cabinet', 'Bathtub', 'Beaker', 'Bear', 
                'Bed', 'Bee', 'Beehive', 'Beer', 'Beetle', 'Bell pepper', 'Belt', 'Bench', 'Bicycle', 
                'Bicycle helmet', 'Bicycle wheel', 'Bidet', 'Billboard', 'Billiard table', 'Binoculars', 
                'Bird', 'Blender', 'Blue jay', 'Bomb', 'Book', 'Bookcase', 'Boot', 'Bottle', 'Bottle opener', 
                'Bow and arrow', 'Bowl', 'Bowling equipment', 'Boy', 'Brassiere', 'Bread', 'Briefcase', 
                'Broccoli', 'Bronze sculpture', 'Brown bear', 'Bull', 'Burrito', 'Bust', 'Butterfly', 
                'Cabbage', 'Cabinetry', 'Cake', 'Cake stand', 'Calculator', 'Camera', 'Can opener', 
                'Canary', 'Candle', 'Candy', 'Cannon', 'Canoe', 'Cantaloupe', 'Carnivore', 'Carrot', 
                'Cart', 'Cassette deck', 'Castle', 'Cat furniture', 'Caterpillar', 'Cattle', 
                'Ceiling fan', 'Cello', 'Centipede', 'Chainsaw', 'Chair', 'Cheese', 'Cheetah', 
                'Chest of drawers', 'Chicken', 'Chime', 'Chisel', 'Chopsticks', 'Christmas tree', 
                'Clock', 'Closet', 'Clothing', 'Cocktail', 'Cocktail shaker', 'Coconut', 'Coffee',
                'Coffee cup', 'Coffee table', 'Coffeemaker', 'Coin', 'Common fig', 'Common sunflower', 
                'Computer keyboard', 'Computer monitor', 'Computer mouse', 'Container', 'Convenience store',
                'Cookie', 'Cooking spray', 'Corded phone', 'Cosmetics', 'Couch', 'Countertop', 'Cowboy hat',
                'Crab', 'Cream', 'Cricket ball', 'Crocodile', 'Croissant', 'Crown', 'Crutch', 'Cucumber',
                'Cupboard', 'Curtain', 'Cutting board', 'Dagger', 'Dairy Product', 'Deer', 'Desk', 'Dessert', 
                'Diaper', 'Dice', 'Digital clock', 'Dinosaur', 'Dishwasher', 'Dog bed', 'Doll', 'Dolphin', 
                'Doughnut', 'Dragonfly', 'Drawer', 'Dress', 'Drill (Tool)', 'Drink', 'Drinking straw', 'Drum', 
                'Duck', 'Dumbbell', 'Eagle', 'Earrings', 'Egg (Food)', 'Elephant', 'Envelope', 'Eraser', 
                'Face powder', 'Facial tissue holder', 'Falcon', 'Fashion accessory', 'Fast food', 'Fax', 
                'Fedora', 'Filing cabinet', 'Fireplace', 'Fish', 'Flag', 'Flashlight', 'Flower', 'Flowerpot', 
                'Flute', 'Flying disc', 'Food', 'Food processor', 'Football', 'Football helmet', 'Footwear', 
                'Fork', 'Fountain', 'Fox', 'French fries', 'French horn', 'Frog', 'Fruit', 'Frying pan', 
                'Furniture', 'Garden Asparagus', 'Gas stove', 'Giraffe', 'Girl', 'Glasses', 'Goat', 
                'Goggles', 'Goldfish', 'Golf ball', 'Golf cart', 'Gondola', 'Goose', 'Grape', 'Grapefruit', 
                'Grinder', 'Guacamole', 'Guitar', 'Hair dryer', 'Hair spray', 'Hamburger', 'Hammer', 'Hamster', 
                'Hand dryer', 'Handbag', 'Handgun', 'Harbor seal', 'Harmonica', 'Harp', 'Harpsichord', 'Hat', 
                'Headphones', 'Heater', 'Hedgehog', 'High heels', 'Hiking equipment', 'Hippopotamus', 'Home appliance', 
                'Honeycomb', 'Horizontal bar', 'Horse', 'Hot dog', 'Houseplant', 'Humidifier', 'Ice cream', 'Indoor rower', 
                'Infant bed', 'Insect', 'Invertebrate', 'Ipod', 'Isopod', 'Jacket', 'Jacuzzi', 'Jaguar (Animal)', 'Jeans', 
                'Jellyfish', 'Jet ski', 'Jug', 'Juice', 'Kangaroo', 'Kettle', 'Kitchen & dining room table', 'Kitchen appliance', 
                'Kitchen knife', 'Kitchen utensil', 'Kitchenware', 'Kite', 'Knife', 'Koala', 'Ladle', 'Ladybug', 'Lamp', 
                'Lantern', 'Laptop', 'Lavender (Plant)', 'Lemon', 'Leopard', 'Light bulb', 'Light switch', 'Lighthouse', 
                'Lily', 'Lion', 'Lipstick', 'Lizard', 'Lobster', 'Loveseat', 'Luggage and bags', 'Lynx', 'Magpie', 'Mammal', 
                'Mango', 'Maple', 'Maracas', 'Marine invertebrates', 'Marine mammal', 'Measuring cup', 'Microphone', 'Microwave oven', 
                'Milk', 'Miniskirt', 'Mirror', 'Mixer', 'Mixing bowl', 'Mobile phone', 'Monkey', 'Moths and butterflies', 'Motorcycle', 
                'Mouse', 'Muffin', 'Mug', 'Mule', 'Mushroom', 'Musical instrument', 'Musical keyboard', 'Nail (Construction)', 'Necklace', 
                'Nightstand', 'Oboe', 'Office building', 'Office supplies', 'Orange', 'Organ (Musical Instrument)', 'Ostrich', 'Otter', 
                'Oven', 'Oyster', 'Paddle', 'Pancake', 'Panda', 'Paper cutter', 'Paper towel', 'Parrot', 'Pasta', 'Pastry', 'Peach', 
                'Pear', 'Pen', 'Pencil case', 'Pencil sharpener', 'Penguin', 'Perfume', 'Personal care', 'Personal flotation device', 
                'Piano', 'Picnic basket', 'Picture frame', 'Pig', 'Pillow', 'Pineapple', 'Pitcher (Container)', 'Pizza', 'Pizza cutter', 
                'Plant', 'Plastic bag', 'Plate', 'Platter', 'Plumbing fixture', 'Polar bear', 'Pomegranate', 'Popcorn', 'Porch', 'Porcupine', 
                'Poster', 'Potato', 'Power plugs and sockets', 'Pressure cooker', 'Pretzel', 'Printer', 'Pumpkin', 'Punching bag', 
                'Racket', 'Radish', 'Ratchet (Device)', 'Rays and skates', 'Red panda', 'Refrigerator', 'Remote control', 'Reptile', 
                'Rifle', 'Ring binder', 'Roller skates', 'Rose', 'Rugby ball', 'Ruler', 'Salad', 'Salt and pepper shakers', 'Sandal', 
                'Sandwich', 'Saucer', 'Saxophone', 'Scale', 'Scarf', 'Scissors', 'Scoreboard', 'Scorpion', 'Screwdriver', 'Sculpture', 
                'Sea lion', 'Sea turtle', 'Seafood', 'Seahorse', 'Seat belt', 'Segway', 'Serving tray', 'Sewing machine', 'Shark', 
                'Sheep', 'Shelf', 'Shellfish', 'Shotgun', 'Shower', 'Shrimp', 'Sink', 'Skateboard', 'Ski', 'Skull', 'Skunk',
                'Skyscraper', 'Slow cooker', 'Snack', 'Snail', 'Snake', 'Snowboard', 'Snowman', 'Snowmobile', 'Snowplow', 
                'Soap dispenser', 'Sock', 'Sofa bed', 'Sombrero', 'Sparrow', 'Spatula', 'Spice rack', 'Spider', 'Spoon', 
                'Sports equipment', 'Sports uniform', 'Squash (Plant)', 'Squid', 'Squirrel', 'Stairs', 'Stapler', 'Starfish', 
                'Stethoscope', 'Stool', 'Strawberry', 'Stretcher', 'Studio couch', 'Submarine', 'Submarine sandwich', 'Suit', 
                'Suitcase', 'Sun hat', 'Sunglasses', 'Surfboard', 'Sushi', 'Swan', 'Swim cap', 'Swimming pool', 'Swimwear', 
                'Sword', 'Syringe', 'Table', 'Table tennis racket', 'Tablet computer', 'Tableware', 'Taco', 'Tap', 'Tart',
                'Tea', 'Teapot', 'Teddy bear', 'Telephone', 'Television', 'Tennis ball', 'Tennis racket', 'Tent', 'Tiara', 
                'Tick', 'Tie', 'Tiger', 'Tin can', 'Tire', 'Toaster', 'Toilet', 'Toilet paper', 'Tomato', 'Tool', 'Toothbrush', 
                'Torch', 'Tortoise', 'Towel', 'Tower', 'Toy', 'Traffic light', 'Traffic sign', 'Train', 'Training bench', 
                'Treadmill', 'Tree house', 'Tripod', 'Trombone', 'Trousers', 'Trumpet', 'Turkey', 'Turtle', 'Umbrella', 
                'Unicycle', 'Vase', 'Vegetable', 'Violin', 'Volleyball (Ball)', 'Waffle', 'Waffle iron', 'Wall clock', 
                'Wardrobe', 'Washing machine', 'Waste container', 'Watch', 'Watercraft', 'Watermelon', 'Weapon', 'Whale', 
                'Whisk', 'Whiteboard', 'Willow', 'Window', 'Window blind', 'Wine', 'Wine glass', 'Wine rack', 'Winter melon', 
                'Wok', 'Wood-burning stove', 'Woodpecker', 'Worm', 'Wrench', 'Zebra', 'Zucchini'
            ]]
        return list(OpenImagesDetection.A1Classes)

    def getId(self,str: str):
        import sys
        if OpenImagesDetection.A1Classes is None:
            OpenImagesDetection.classesList()
        if str in OpenImagesDetection.A1Classes:
            return OpenImagesDetection.A1Classes.index(str)
        else:
            #print(str, "is not a known category from OpenImages", file=sys.stderr)
            return OpenImagesDetection.getId("void")

    def getName(self,id=None):
        if OpenImagesDetection.A1Classes is None:
            OpenImagesDetection.classesList()
        if id is None or isinstance(id, OpenImagesDetection):
            return "OpenImages"
        if id >= 0 and id < len(OpenImagesDetection.A1Classes):
            return OpenImagesDetection.A1Classes[id]
        return "void"

    def isBanned(self,nameOrId):
        if OpenImagesDetection.A1Classes is None:
            OpenImagesDetection.classesList()
        if isinstance(nameOrId, str):
            return nameOrId == "void"
        else:
            return OpenImagesDetection.isBanned(OpenImagesDetection.getName(nameOrId))

    images: fo.Dataset
    n: int

    def __init__(self, dataset) -> None:
        self.images = dataset
        self.n = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> Sample:

        if isinstance(index,slice):
            values=[]
            if index.step is not None:
                values = [v for v in range(index.start,index.stop,index.step)]
            else:
                values = [v for v in range(index.start,index.stop)]
            values = [v for v in values if v < len(self.images)]
            if len(values)==0:
                raise StopIteration
            return [self.__getitem__(v) for v in values]
        else:
            if self.n is None:
                self.n = self.images.__iter__()
            return self.__next__()

    def __iter__(self):
        self.n = self.images.__iter__()
        return self

    def __next__(self) -> Sample:
        value = self.n.__next__()
        citiSamp = Sample.fromFiftyOne(value)
        dict = value.to_dict()

        citiSamp.detection = Detection()
        for d in dict["detections"]["detections"]:
            box = Box2d()
            box.x = d["bounding_box"][0] * citiSamp.getRGB().shape[2]
            box.y = d["bounding_box"][1] * citiSamp.getRGB().shape[1]
            box.w = d["bounding_box"][2] * citiSamp.getRGB().shape[2]
            box.h = d["bounding_box"][3] * citiSamp.getRGB().shape[1]
            box.c = OpenImagesDetection.getId(d["label"])
            box.cn = d["label"]
            if not OpenImagesDetection.isBanned(d["label"]):
                citiSamp.detection.boxes2d.append(box)
            #print(d)

        return citiSamp
