import pygame
import random
import neat
import os


#PNG import 

BIRD_IMG = pygame.image.load("bird.png")
PIPE_IMG = pygame.image.load("pipe.png")
BG_IMG = pygame.image.load("sky.png")


# Initialize pygame
pygame.init()


#score font:
FONT = pygame.font.SysFont("arial", 40)

# Window settings
WIDTH = 500
HEIGHT = 700
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simple Flappy Bird")


BIRD_IMG = pygame.transform.scale(BIRD_IMG, (40,40))
PIPE_IMG = pygame.transform.scale(PIPE_IMG, (70,500))
BG_IMG = pygame.transform.scale(BG_IMG, (WIDTH, HEIGHT))

# Game constants
GRAVITY = 0.5
JUMP_STRENGTH = -10
PIPE_WIDTH = 70
PIPE_GAP = 200
PIPE_VELOCITY = 5

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
BLUE = (135, 206, 235)
YELLOW = (255, 255, 0)


# ---------------------------
# Bird Class
# ---------------------------
class Bird:
    def __init__(self, x, y):
        # Initial position of bird
        self.x = x
        self.y = y

        # Bird physics variables
        self.vel = 0

        # Bird size
        self.width = 30
        self.height = 30

    def jump(self):
        """
        Makes the bird move upward.
        We give it a negative velocity to move up.
        """
        self.vel = JUMP_STRENGTH

    def move(self):
        """
        Updates the bird position every frame.
        Gravity slowly pulls the bird downward.
        """
        self.vel += GRAVITY
        self.y += self.vel

    def draw(self, win):
        """
        Draws the bird on the screen.
        """
        # pygame.draw.rect(win, YELLOW, (self.x, self.y, self.width, self.height))

        win.blit(BIRD_IMG, (self.x, self.y))


# ---------------------------
# Pipe Class
# ---------------------------
class Pipe:
    def __init__(self, x):
        # Starting x position of pipe
        self.x = x

        # Random height of pipe
        self.height = random.randint(50, 400)

        # Calculate positions of top and bottom pipes
        self.top = self.height
        self.bottom = self.height + PIPE_GAP

        self.width = PIPE_WIDTH

        self.passed = False

    def move(self):
        """
        Moves the pipe to the left every frame.
        """
        self.x -= PIPE_VELOCITY

    def draw(self, win):
        """
        Draws the top and bottom pipes.
        """
        # # Top pipe
        # pygame.draw.rect(win, GREEN, (self.x, 0, self.width, self.top))

        # # Bottom pipe
        # pygame.draw.rect(win, GREEN, (self.x, self.bottom, self.width, HEIGHT))

        win.blit(PIPE_IMG, (self.x, self.bottom))
        win.blit(pygame.transform.flip(PIPE_IMG, False, True), (self.x, self.top - PIPE_IMG.get_height()))


# ---------------------------
# Draw Window Function
# ---------------------------
# def draw_window(win, bird, pipes):
#     """
#     Draws everything on the screen.
#     """
#     win.fill(BLUE)

#     # Draw bird
#     bird.draw(win)

#     # Draw pipes
#     for pipe in pipes:
#         pipe.draw(win)

#     pygame.display.update()

# def draw_window(win, birds, pipes):

#     win.fill((135,206,235))

#     for pipe in pipes:
#         pipe.draw(win)

#     for bird in birds:
#         bird.draw(win)

#     pygame.display.update()


def draw_window(win, birds, pipes, score):

    win.blit(BG_IMG, (0,0))

    for pipe in pipes:
        pipe.draw(win)

    for bird in birds:
        bird.draw(win)

    score_text = FONT.render(f"Score: {score}", True, (255,255,255))
    win.blit(score_text, (10,10))

    pygame.display.update()


# ---------------------------
# Main Game Function
# ---------------------------
# def main():
#     run = True
#     clock = pygame.time.Clock()

#     bird = Bird(200, 300)

#     # List of pipes
#     pipes = [Pipe(500)]

#     while run:
#         clock.tick(30)  # FPS

#         # Event handling
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 run = False

#             # Space key makes bird jump
#             if event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_SPACE:
#                     bird.jump()

#         # Move bird
#         bird.move()

#         # Move pipes
#         for pipe in pipes:
#             pipe.move()

#             # Spawn new pipe when one goes halfway
#             if pipe.x < 250 and len(pipes) < 2:
#                 pipes.append(Pipe(500))

#             # Remove pipe when it leaves screen
#             if pipe.x + pipe.width < 0:
#                 pipes.remove(pipe)

#         # Draw everything
#         draw_window(WIN, bird, pipes)

#     pygame.quit()

# ----------------
# NEAT comes in the story 
#-----------------

def eval_genomes(genomes, config):
    birds = []
    nets = []
    ge = []

    # Create a bird for every genome
    for genome_id, genome in genomes:

        genome.fitness = 0  # start fitness

        net = neat.nn.FeedForwardNetwork.create(genome, config)

        nets.append(net)
        birds.append(Bird(200, 300))
        ge.append(genome)

    pipes = [Pipe(500)]
    score = 0

    run = True
    clock = pygame.time.Clock()

    while run and len(birds) > 0:

        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].width:
                pipe_ind = 1

        for x, bird in enumerate(birds):

            bird.move()

            # increase fitness the longer the bird survives
            ge[x].fitness += 0.1

            pipe = pipes[pipe_ind]

            # Inputs to neural network
            inputs = (
                bird.y,
                abs(bird.y - pipe.height),
                abs(bird.y - pipe.bottom)
            )

            output = nets[x].activate(inputs)

            # If output > 0.5 → jump
            if output[0] > 0.5:
                bird.jump()

        add_pipe = False
        rem = []

        for pipe in pipes:

            pipe.move()

            for x, bird in enumerate(birds):

                # Collision detection
                if (
                    bird.x + bird.width > pipe.x
                    and bird.x < pipe.x + pipe.width
                ):

                    if bird.y < pipe.top or bird.y + bird.height > pipe.bottom:

                        ge[x].fitness -= 1
                        birds.pop(x)
                        nets.pop(x)
                        ge.pop(x)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True
                    score +=1

            if pipe.x + pipe.width < 0:
                rem.append(pipe)

        if add_pipe:

            for genome in ge:
                genome.fitness += 5

            pipes.append(Pipe(500))

        for r in rem:
            pipes.remove(r)

        for x, bird in enumerate(birds):

            if bird.y + bird.height >= HEIGHT or bird.y < 0:
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)

        draw_window(WIN, birds, pipes, score)


def run(config_path):

    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))

    winner = population.run(eval_genomes, 50)


if __name__ == "__main__":

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")

    run(config_path)

# Run the game
# main()