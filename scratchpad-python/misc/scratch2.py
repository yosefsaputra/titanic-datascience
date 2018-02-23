def subtractChar(char, int_x):
    '''
    This function substracts a character ordinal value with an integer x.
    However, it preserves the uppercase or lowercase state of the character
    :param char: single alphabet character
    :type char: str
    :param int_x: an integer number to substract
    :type int_x: int
    :return: result
    '''
    result = char
    if (97 <= ord(char) <= 122):
        result = ord(char) - int(int_x)
        if result < 97:
            result += 26
    elif (65 <= ord(char) <= 90):
        result = ord(char) - int(int_x)
        if result < 65:
            result += 26

    return chr(result)


def getKey(key, int_x):
    '''
    This function returns the int_x(th) character in key. Rotation is applied.
    :param key: the key to be extracted
    :type key: str
    :param int_x: an integer number that indicates the location of character in key to be returned
    :type int_x: int
    :return: str
    '''
    n = len(key)
    int_x = int(int_x) % n
    return key[int_x]


def decrypt(encrypted_message):
    '''
    This function decrypts the message from Alice according to a specific key.
    :param encrypted_message: the message to be decrypted
    :type encrypted_message: str
    :return: str
    '''
    key = '8251220'
    keyPointer = 0

    decrypted_message = ''

    for char in encrypted_message:
        if char.isalpha():
            decrypted_message += subtractChar(char, int(getKey(key, keyPointer)))
            keyPointer += 1
        else:
            decrypted_message += char

    return decrypted_message