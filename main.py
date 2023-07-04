import tkinter as tk
import tkinter.messagebox
import webbrowser
import yfinance as yf
import requests
import re
import ai
from bs4 import BeautifulSoup
import json
import TopScraper
from tkinter import ttk
from datetime import datetime, timedelta
index_var = None
def output(accuracy, arr):
    root.title("Displaying Predictiong - Profit/Loss")
    # Create a canvas and add a scrollbar to it
    welcome_label = tk.Label(root, text="Prediction with accuracy " + str(int(accuracy*100)) + "%")
    welcome_label.pack(pady=10)
    canvas = tk.Canvas(root, height=500)
    scrollbar = ttk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.config(yscrollcommand=scrollbar.set)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Create a frame inside the canvas for the table
    table_frame = tk.Frame(canvas)
    table_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    table_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    # Get today's date
    today = datetime.today()

    # Loop over the array and create labels for each element
    for i, val in enumerate(arr):
        # Calculate the date i days from now
        date = (today + timedelta(days=i+1)).strftime('%m/%d/%Y')

        # Create the label for the "After i days" column
        days_label = tk.Label(table_frame, text=f"{date}\nAfter {i+1} days", font=("Arial", 12, "bold"), borderwidth=1, relief="solid", justify="center")
        days_label.grid(row=i, column=0, padx=5, pady=5, sticky="nsew")

        # Create the label for the "PROFIT/LOSS" column
        if val == 1:
            pl_label = tk.Label(table_frame, text="PROFIT", font=("Arial", 12, "bold"), bg="green", fg="white", borderwidth=1, relief="solid")
        else:
            pl_label = tk.Label(table_frame, text="LOSS", font=("Arial", 12, "bold"), bg="red", fg="white", borderwidth=1, relief="solid")
        pl_label.grid(row=i, column=1, padx=5, pady=5, sticky="nsew")

        # Set column weight for both columns
        table_frame.columnconfigure(0, weight=1)
        table_frame.columnconfigure(1, weight=1)

    # Center the table frame horizontally
    canvas.create_window((0, 0), window=table_frame, anchor="nw")
    table_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    root.update()
    canvas_width = canvas.winfo_width()
    canvas.create_window((canvas_width/2, 0), window=table_frame, anchor="n")
def result(numdays=100, interval = '1d', symbol = 'GOOG', stationarity = False, window = 15):
    model = ai.SMP(int(numdays),interval, symbol,bool(stationarity), int(window))
    model.outliers()
    model.makeStationary()
    model.smoothData()
    model.getIndicatorData()
    model.producePrediction()
    model.getModelAccuracy()
    accuracy, df = model.makePrediction()
    print(df)
    output(accuracy, df)
def predictor(index):
    """
    This function predicts the stock market index based on the given index.
    """
    # Create a frame to hold the input fields and buttons
    frame = tk.Frame(root,bg='',highlightthickness=0, bd=0)
    frame.place(relheight=1, relwidth=1, relx=0.5,rely=0.5, anchor='center')
    root.title("Stock Market Predictor")
    bg_label = tk.Label(frame,image=background_image)
    bg_label.place(relheight=1, relwidth=1)
    header_label1 = tk.Label(frame, text="Please enter parameters for prediction:", font=("Arial",12))
    header_label1.place(relx=0.5, rely=0.25, anchor="center")
    window_label = tk.Label(frame, text="Window (interval) for prediction:")
    window_label.place(relx=0.1, rely=0.45)
    window_label_var = tk.StringVar()
    window_label_entry = tk.Entry(frame, textvariable=index_var, highlightthickness=0, highlightbackground="black")
    window_label_entry.place(relx=0.5, rely=0.45)
    days_label = tk.Label(frame, text="Number of days for dataset:")
    days_label.place(relx=0.1, rely=0.55)
    days_var = tk.StringVar()
    days_entry = tk.Entry(frame, textvariable=index_var, highlightthickness=0, highlightbackground="black")
    days_entry.place(relx=0.5, rely=0.55)
    # create a variable to store the user's selection
    selection = tk.StringVar(value="False")

    # create radio buttons for the options
    stationary_button = tk.Radiobutton(frame, text="Stationary", variable=selection, value="True")
    stationary_button.place(relx=0.5, rely=0.6)

    non_stationary_button = tk.Radiobutton(frame, text="Non-Stationary", variable=selection, value="False")
    non_stationary_button.place(relx=0.5, rely=0.65)
    submit_button = tk.Button(frame, text="Submit", command=lambda:[result(days_entry.get(), '1d', index,  selection.get(), window_label_entry.get()), frame.destroy()])
    submit_button.place(relx=0.5, rely=0.75)

def get_ticker_symbol(company_name):
    url = f"https://finance.yahoo.com/_finance_doubledown/api/resource/searchassist;searchTerm={company_name}"
    response = requests.get(url, headers={'User-Agent': 'Custom'})
    if response.status_code == 200:
        data = json.loads(response.text)
        if data["items"]:
            return data["items"][0]["symbol"]
        else:
            return ""
    else:
        return ""

def check_valid_index(index):
    """
    This function checks if the stock market index is valid.
    If the index is not valid, it suggests a similar index if available.
    """
    tickers = yf.Tickers([index])
    if index in tickers.tickers:
        return index
    else:
        company = get_ticker_symbol(index)
        if(company!=""):
            response = tkinter.messagebox.askyesno("Stock Market Index Error", f"Did you mean {company}, the stock market index for {index}?")
            if(response):
                return company
        return ""

def submit_index(frame, index):
    """
    This function is called when the "Submit" button is clicked
    """
    index = check_valid_index(index)
    if index!="":
        predictor(index)
        frame.destroy()
    else:
        tkinter.messagebox.showerror("Stock Market Index Error", "Invalid Stock Market Index! Please search online")

def search_online(index):
    """
    This function is called when the "Search Online" button is clicked
    """
    url = f"https://www.google.com/search?q=what+is+the+stock+market+index+for+{index}"
    webbrowser.open(url)

def my_disp(window, df): # display the Treeview with data
    l1=list(df) # List of column names as headers 
    r_set=df.to_numpy().tolist() # Create list of list using rows 
    trv = ttk.Treeview(window, selectmode ='browse',
                       show='headings',height=10,columns=l1)
    trv.grid(row=2,column=1,columnspan=4,padx=15,pady=10)
    for col in l1:
        trv.column(col, width = 100, anchor ='w')
        trv.heading(col, text =col,command=lambda col=col :my_sort(col, window, df))

    ## Adding data to treeview 
    for dt in r_set:  
        v=[r for r in dt] # creating a list from each row 
        trv.insert("",'end',iid=v[0],values=v) # adding row
    vs = ttk.Scrollbar(window,orient="vertical", command=trv.yview)#V Scrollbar
    trv.configure(yscrollcommand=vs.set)  # connect to Treeview
    vs.grid(row=2,column=5,sticky='ns')
def my_sort(col, window, df): # Update the dataframe after sorting
    global order 
    if order:
        order=False # set ascending value
    else:
        order=True
    df=df.sort_values(by=[col],ascending=order)
    my_disp(window, df) # refresh the Treeview 
def getTrending():
    window = tk.Toplevel()
    window.geometry("600x300")
    window.title("Today's Trending Tickers")
    df = TopScraper.getDF()
    my_disp(window, df)
def getIndex(root):
    # Create a frame to hold the input fields and buttons
    frame = tk.Frame(root,bg='',highlightthickness=0, bd=0)
    frame.place(relheight=1, relwidth=1, relx=0.5,rely=0.5, anchor='center')
    bg_label = tk.Label(frame,image=background_image)
    bg_label.place(relheight=1, relwidth=1)
    header_label1 = tk.Label(frame, text="Welcome! Please enter the stock market index:", font=("Arial",12))
    header_label1.place(relx=0.5, rely=0.25, anchor="center")
    # Create the input fields and button
    index_label = tk.Label(frame, text="Stock Market Index:")
    index_label.place(relx=0.1, rely=0.45)
    index_var = tk.StringVar()
    index_entry = tk.Entry(frame, textvariable=index_var, highlightthickness=0, highlightbackground="black")
    index_entry.place(relx=0.4, rely=0.45)

    submit_button = tk.Button(frame, text="Submit", command=lambda:submit_index(frame, index_entry.get()))
    submit_button.place(relx=0.4, rely=0.55)

    search_button = tk.Button(frame, text="Search Online", command=lambda:search_online(index_entry.get()))
    search_button.place(relx=0.5, rely=0.55)
    
    tickers_button = tk.Button(frame,text = "Get Today's Trending", command = getTrending)
    tickers_button.place(relx=0.4, rely=0.7)
# Create the main window
root = tk.Tk()
root.title("Stock Market Predictor")
root.geometry("500x500")

# Set the background image
background_image = tk.PhotoImage(file="bg.ppm")
background_label = tk.Label(root, image=background_image)
background_label.place(relwidth=1, relheight=1)

# Create the frame and input fields
getIndex(root)
order = True
# Start the main event loop
root.mainloop()
quit()
