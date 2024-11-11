import pygame
import numpy as np
import cv2
import json
import random
import subprocess
pygame.init()

canvas_width = 280  #Width of the canvas
canvas_height = 280  #Height of the canvas
panel_width = 150  #Width of info area
window_width = canvas_width + panel_width
window_height = canvas_height

screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption('Draw a Number')

#Set background and colors
white = (255, 255, 255)
black = (0, 0, 0)
gray = (200, 200, 200)
drawrad = 20

screen.fill(white)
pygame.draw.rect(screen, gray, (canvas_width, 0, panel_width, window_height))

running = True
drawing = False  
last_pos = None

def display_text(text, x, y, font_size=20):
    font = pygame.font.SysFont('arial', font_size)
    text_surface = font.render(text, True, black)
    screen.blit(text_surface, (x, y))


def image_data_to_grayscale(img_data):
    height, width, _ = img_data.shape
    grayscale_img = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            # rgb
            grayscale_value = img_data[y, x, 0] / 255.0
            grayscale_img[y, x] = grayscale_value

    return grayscale_img

def get_bounding_rectangle(img, threshold):
    rows, columns = img.shape
    min_x, min_y = columns, rows
    max_x, max_y = -1, -1

    for y in range(rows):
        for x in range(columns):
            if img[y, x] < threshold:
                if min_x > x:
                    min_x = x
                if max_x < x:
                    max_x = x
                if min_y > y:
                    min_y = y
                if max_y < y:
                    max_y = y

    return {'minY': min_y, 'minX': min_x, 'maxY': max_y, 'maxX': max_x}

def center_image(img):
    mean_x = 0
    mean_y = 0
    rows, columns = img.shape
    sum_pixels = 0

    for y in range(rows):
        for x in range(columns):
            pixel = 1 - img[y, x]
            sum_pixels += pixel
            mean_y += y * pixel
            mean_x += x * pixel

    if sum_pixels == 0:
        return {'transX': 0, 'transY': 0}

    mean_x /= sum_pixels
    mean_y /= sum_pixels

    d_y = round(rows / 2 - mean_y)
    d_x = round(columns / 2 - mean_x)

    return {'transX': d_x, 'transY': d_y}

def preprocess_and_center_image(pygame_image):
    gray_image = image_data_to_grayscale(np.transpose(pygame_image, (1, 0, 2)))

    bounding_rect = get_bounding_rectangle(gray_image, threshold=0.5)

    min_y, min_x, max_y, max_x = bounding_rect['minY'], bounding_rect['minX'], bounding_rect['maxY'], bounding_rect['maxX']
    if min_x < max_x and min_y < max_y:
        cropped_image = gray_image[min_y:max_y+1, min_x:max_x+1]

        # Maintain aspect ratio while resizing
        h, w = cropped_image.shape
        aspect_ratio = max(h, w)
        scaling_factor = 20.0 / aspect_ratio
        new_h = int(h * scaling_factor)
        new_w = int(w * scaling_factor)

        digit_resized = cv2.resize(cropped_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create 28x28 canvas
        centered_image = np.full((28, 28), 1.0, dtype=np.float64)
        offset_x = (28 - new_w) // 2
        offset_y = (28 - new_h) // 2
        centered_image[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = digit_resized

    
        centered_image = 1.0 - centered_image

        # TMP visual
        # cv2.imshow('Centered Image', centered_image * 255)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()

        cv2.imwrite('drawn_digit.png', centered_image*255)
        display_text("Image saved!", canvas_width + 10, 120)
        print("Image saved as 'drawn_digit.png'")

        return centered_image  
    else:
        return np.full((28, 28), 255, dtype=np.float64)




currentVal = -1
path = "./net.exe"
neural = subprocess.Popen(
    [path],  # The program path and arguments if needed
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,  # Ensures input/output are handled as strings
)
def send_command(command):
    if neural.stdin and neural.stdout:
        neural.stdin.write(command + '\n')
        neural.stdin.flush()  # Ensure the command is sent immediately
        response = neural.stdout.readline()  # Read response line-by-line
        return response.strip()



def evalImage(img_data):
    global currentVal
    normData = (img_data).flatten().tolist()
    for x in range(len(normData)):
       
        normData[x] =  max(0,round(normData[x],12))

    print(json.dumps(normData))
    RES = send_command("eval " + json.dumps(normData))
    RES = json.loads(RES.replace(" ", ","))
    ##print(RES)
    mxVal = 0
    mxIndex = -1
    for x in range(len(RES)):
        if RES[x] > mxVal:
            mxVal = RES[x]
            mxIndex = x
    currentVal = mxIndex


print(send_command("load net.json"))

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.pos[0] < canvas_width:
                drawing = True
                last_pos = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False 
            last_pos = None

            pygame_image = pygame.surfarray.array3d(screen.subsurface((0, 0, canvas_width, canvas_height)))
            processed_image = preprocess_and_center_image(pygame_image)

            evalImage(processed_image)

        
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:  # 'C'
                screen.fill(white, (0, 0, canvas_width, canvas_height))
                display_text("Canvas cleared", canvas_width + 10, 150)

   
    if drawing:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        if mouse_x < canvas_width:
            if last_pos is not None:
                
                pygame.draw.line(screen, black, last_pos, (mouse_x, mouse_y), drawrad) #DrawWidth
            last_pos = (mouse_x, mouse_y) 
            pygame.draw.circle(screen, black,(mouse_x, mouse_y),drawrad/2)


    #INFO area
    pygame.draw.rect(screen, gray, (canvas_width, 0, panel_width, window_height))
    display_text("Press 'C' to clear", canvas_width + 10, 30)
    display_text("Last number:" + str(currentVal), canvas_width + 10, 90)

    # Update the display
    pygame.display.flip()

send_command("exit")
pygame.quit()
