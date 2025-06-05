import pygame
from pygame.math import Vector2

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((1000, 1000))
clock = pygame.time.Clock()

# Load and scale obstacle image
image_path = "Assignment_0/images/triangle@50px.png"
obstacle_image = pygame.image.load(image_path).convert_alpha()
obstacle_image = pygame.transform.scale(obstacle_image, (50, 50))
obstacle_rect = obstacle_image.get_rect()

# Obstacle position
obstacle_pos = Vector2(500, 500)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))  # Black background

    # Draw the obstacle image
    draw_pos = (int(obstacle_pos.x - obstacle_rect.width / 2),
                int(obstacle_pos.y - obstacle_rect.height / 2))
    screen.blit(obstacle_image, draw_pos)

    # Debug red dot
    pygame.draw.circle(screen, (255, 0, 0), (int(obstacle_pos.x), int(obstacle_pos.y)), 4)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
