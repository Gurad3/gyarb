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

screen.fill(white)
pygame.draw.rect(screen, gray, (canvas_width, 0, panel_width, window_height))

running = True
drawing = False  
last_pos = None

def display_text(text, x, y, font_size=20):
    font = pygame.font.SysFont('arial', font_size)
    text_surface = font.render(text, True, black)
    screen.blit(text_surface, (x, y))

def preprocess_and_center_image(pygame_image):
    #Convert to grayscale and find bounding box
    gray_image = cv2.cvtColor(np.transpose(pygame_image, (1, 0, 2)), cv2.COLOR_RGB2GRAY)
    _, thresholded = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
    
    
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        cropped_image = gray_image[y:y+h, x:x+w]
        
        #Resize to 20x20 
        digit_resized = cv2.resize(cropped_image, (20, 20), interpolation=cv2.INTER_AREA)
        
        #Create  28x28 canvas and center
        centered_image = np.full((28, 28), 255, dtype=np.float64)
        offset_x = (28 - 20) // 2
        offset_y = (28 - 20) // 2
        centered_image[offset_y:offset_y+20, offset_x:offset_x+20] = digit_resized
        
       
        #TMP VISUAL
        cv2.imshow('Centered Image', centered_image)
        cv2.waitKey(1000) #(ms)
        cv2.destroyAllWindows()
      
        for y in range(len(centered_image)):
            for x in range(len(centered_image[y])):
                centered_image[x][y] = (255 - centered_image[x][y])

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

print(send_command("load net.json"))


def evalImage(img_data):
    global currentVal
    normData =  (img_data / 255).flatten().tolist()

    #print(json.dumps(normData))
    RES = send_command("eval " + json.dumps(normData))
    RES = json.loads(RES.replace(" ", ","))
    print(RES)
    mxVal = 0
    mxIndex = -1
    for x in range(len(RES)):
        if RES[x] > mxVal:
            mxVal = RES[x]
            mxIndex = x
    currentVal = mxIndex

    



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
                
                pygame.draw.line(screen, black, last_pos, (mouse_x, mouse_y), 20) #DrawWidth
            last_pos = (mouse_x, mouse_y) 


    #INFO area
    pygame.draw.rect(screen, gray, (canvas_width, 0, panel_width, window_height))
    display_text("Press 'C' to clear", canvas_width + 10, 30)
    display_text("Last number:" + str(currentVal), canvas_width + 10, 90)

    # Update the display
    pygame.display.flip()

send_command("exit")
pygame.quit()
