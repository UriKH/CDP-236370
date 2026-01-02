#
#   @date:  [TODO: Today's date]
#   @author: [TODO: Student Names]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2025
#
from multiprocessing import Lock, Pipe


class MyQueue(object):

    def __init__(self):
        ''' Initialize MyQueue and it's members.
        '''
        self.lock = Lock()
        self.reader, self.writer = Pipe()

    def put(self, msg):
        '''Put the given message in queue.

        Parameters
        ----------
        msg : object
            the message to put.
        '''
        self.lock.acquire()
        self.writer.send(msg)
        self.lock.release()

    def get(self):
        '''Get the next message from queue (FIFO)
            
        Return
        ------
        An object
        '''
        return self.reader.recv()

    def empty(self):
        '''Get whether the queue is currently empty
            
        Return
        ------
        A boolean value
        '''
        return not self.reader.poll()
