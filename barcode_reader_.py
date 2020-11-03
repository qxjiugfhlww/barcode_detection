import cv2

​
def read_barcodes(frame):
    barcodes = pyzbar.decode(frame)
    for barcode in barcodes:
        x, y , w, h = barcode.rect
        barcode_text = barcode.data.decode('utf-8')
        print(barcode_text)
        cv2.rectangle(frame, (x, y),(x+w, y+h), (0, 255, 0), 2)
    return frame
​
def main():
    
    frame = cv2.imread('photo/90.bmp')
    frame = read_barcodes(frame)
    cv2.imshow('Barcode reader', frame)


​
if __name__ == '__main__':
    main()