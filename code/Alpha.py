#!/usr/bin/env python
# coding: utf-8

# # Libraries and Dependencies

# In[1]:


import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.subplots as sp
import pandas as pd
import numpy as np


# # The Data

# In[2]:


df_raw = pd.read_csv('Individual risk preference - Form Responses 1.csv')
df = df_raw.copy()


# In[3]:


df.columns


# ## Year

# In[4]:


# Convert the timestamp column to a datetime object
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Group the data by year and count the number of responses for each year
yearly_counts = df.groupby(df['Timestamp'].dt.year).count()['Timestamp']

# Create a line plot of the yearly response counts using Plotly's graph objects
fig = go.Figure()

# Add a trace for the response counts
fig.add_trace(
    go.Scatter(
        x=yearly_counts.index,
        y=yearly_counts.values,
        text=yearly_counts.values, # Add value labels
        mode='lines+markers+text', # Add text labels to markers
        textposition='top center', # Center text labels on markers
        line=dict(color='blue', width=2, shape='spline'),
        marker=dict(color='blue', size=8),
        name='Response Counts'
    )
)


# Set the plot title and axis labels
fig.update_layout(
    title='Number of Responses Over Time',
    xaxis_title='Year',
    yaxis_title='Response Count',
    plot_bgcolor='rgba(0,0,0,0)', # Set transparent background
    title_x=0.5, # Center the title horizontally
    title_y=0.9, # Center the title vertically
)

# Display the plot
fig.show()


# ## Gender

# In[5]:




# Count the number of responses for each gender
gender_counts = df['Какъв е Вашият пол?'].replace("Жена / Female", "Female").replace("Мъж / Male","Male").value_counts()

# Calculate the percentage split between two categories
split_percent = round(gender_counts[0] / gender_counts[1], 2)
split_label = f'1x:{split_percent}x'

# Create a Plotly donut chart of the gender counts with a split label in the center
fig = go.Figure(data=[go.Pie(
    labels=gender_counts.index,
    values=gender_counts.values,
    hole=0.6, # Set the size of the donut hole
    marker=dict(colors=['#FFC0CB', '#87CEFA']), # Set the colors of the pie slices
    textinfo='percent', # Set the text format for the labels
)])

# Add a text label in the center of the donut chart to display the split between two categories
fig.add_annotation(
    x=0.5,
    y=0.5,
    text=split_label,
    font=dict(size=30),
    showarrow=False
)

# Set the title and legend position of the chart
fig.update_layout(
    title={
        'text': 'Gender Distribution',
        'y': 0.95, # Set the vertical position of the title
        'x': 0.5, # Center the title horizontally
        'xanchor': 'center', # Set the anchor point of the title
        'yanchor': 'top' # Set the anchor point of the title
    },
    legend={
        'orientation': 'h', # Set the orientation of the legend to horizontal
        'y': 1.15, # Set the distance between the legend and the chart
        'x': 0.5, # Center the legend horizontally
        'xanchor': 'center', # Set the anchor point of the legend
        'yanchor': 'top', # Set the anchor point of the legend
        'traceorder': 'reversed' # Reverse the order of the legend items
    }
)

# Display the chart
fig.show()


# ## Age

# In[6]:


# Define the age buckets and corresponding labels
age_buckets = [0, 15,20, 25, 30, 35, 45, 55, 65]
age_labels = [f'{a}-{b} years' for a, b in zip(age_buckets[:-1], age_buckets[1:])]

# Group the data by age bucket and count the number of responses for each bucket
age_counts = pd.cut(df['Каква е Вашата възраст в години?'], bins=age_buckets, include_lowest=True).value_counts(sort=False)

# Create a bar chart of the age counts using Plotly's graph objects
fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=age_labels,
        y=age_counts.values,
        marker_color='blue',
        name='Age Counts',
        text=age_counts.values, # Add value labels
        textposition='outside', # Display value labels on top of the bars
    )
)

# Set the plot title and axis labels
fig.update_layout(
    title='Distribution of Respondent Ages',
    xaxis_title='Age Buckets',
    yaxis_title='Number of Responses',
    plot_bgcolor='rgba(0,0,0,0)', # Set transparent background
    title_x=0.5, # Center the title horizontally
    title_y=0.9, # Center the title vertically
)
fig.update_layout(
    title={
        'text': 'Distribution of Respondent Ages',
        'y': 0.95, # Set the vertical position of the title
        'x': 0.5, # Center the title horizontally
        'xanchor': 'center', # Set the anchor point of the title
        'yanchor': 'top' # Set the anchor point of the title
    },
    legend={
        'orientation': 'h', # Set the orientation of the legend to horizontal
        'y': 1.15, # Set the distance between the legend and the chart
        'x': 0.5, # Center the legend horizontally
        'xanchor': 'center', # Set the anchor point of the legend
        'yanchor': 'top', # Set the anchor point of the legend
        'traceorder': 'reversed' # Reverse the order of the legend items
    })

# Display the plot
fig.show()


# In[7]:


age_counts


# ## Education in years

# In[8]:


# Create a distribution plot of the years of education using Plotly's create_distplot function
fig = ff.create_distplot([df['Колко общо години образование имате от първи клас включително? // How many years of first grade education do you have in total?'].dropna()],
                         ['Years of Education'], colors=['blue'], show_rug=False, curve_type='normal', bin_size = 1)

# Set the plot title and axis labels
fig.update_layout(
    title='Distribution of Respondents by Years of Education',
    xaxis_title='Years of Education',
    yaxis_title='Probability Density',
    plot_bgcolor='rgba(0,0,0,0)', # Set transparent background
    title_x=0.5, # Center the title horizontally
    title_y=0.9, # Center the title vertically,
    legend=dict(x=0.5, y=1.2, orientation='h') 
)
fig.update_layout(
    title={
        
        'y': 0.95, # Set the vertical position of the title
        'x': 0.5, # Center the title horizontally
        'xanchor': 'center', # Set the anchor point of the title
        'yanchor': 'top' # Set the anchor point of the title
    },
    legend={
        'orientation': 'h', # Set the orientation of the legend to horizontal
        'y': 1.15, # Set the distance between the legend and the chart
        'x': 0.5, # Center the legend horizontally
        'xanchor': 'center', # Set the anchor point of the legend
        'yanchor': 'top', # Set the anchor point of the legend
        'traceorder': 'reversed' # Reverse the order of the legend items
    })


# Display the plot
fig.show()


# In[9]:


df['Колко общо години образование имате от първи клас включително? // How many years of first grade education do you have in total?'].dropna().value_counts()


# ## Education areas

# In[10]:


import pandas as pd

def group_by_education(df, column_name):
    # Get the value counts for the column
    education_counts = df[column_name].value_counts()

    # Determine the categories that have less than 5 counts
    other_categories = education_counts[education_counts < 7].index.tolist()

    # Group the responses into the desired categories, including an "Other" category
    df[column_name] = df[column_name].apply(lambda x: x if x not in other_categories else 'Other')

    # Recalculate the education counts with the new groupings
    education_counts = df[column_name].value_counts()

    # Return the education counts
    return education_counts


# In[11]:


df['Education'] = df['В какво професионално направление е Вашето най-съществено образование?']
df['Education'] = df['Education'].replace('Икономика и бизнес  //  Economics and Business', 'Economics and Business')
df['Education'] = df['Education'].replace('Технически науки  //  Technical sciences', 'Technical sciences')
df['Education'] = df['Education'].replace('Информатика и компютърни науки.', 'Informational technologies')
df['Education'] = df['Education'].replace('Хуманитарни науки // Humanities', 'Humanities')
df['Education'] = df['Education'].replace('Социални науки  //  Social Sciences', 'Social Sciences')
df['Education'] = df['Education'].replace('Здравеопазване и спорт', 'Social Sciences')
df['Education'] = df['Education'].replace('Математика  //  Mathematics', 'Technical sciences')
df['Education'] = df['Education'].replace('Изкуства', 'Art')
df['Education'] = df['Education'].replace('Правни и политически науки', 'Law & Political Science')
df['Education'] = df['Education'].replace('Природни науки', 'Environmental Science')
df['Education'] = df['Education'].replace('Бизнес администрация', 'Economics and Business')
df['Education'] = df['Education'].replace('Педагогически науки', 'Pedagogical sciences')
df['Education'] = df['Education'].replace('Администрация и управление', 'Economics and Business')
df['Education'] = df['Education'].replace('Информатика и компютърни науки  //  Informatics and computer science', 'Informational technologies')


# In[12]:


education_counts = group_by_education(df, 'Education')

# Print the education counts
education_counts.values, education_counts.index


# In[13]:


education_counts/825


# In[14]:


education_counts.values.sum()


# In[15]:


# Create the bar chart
fig = go.Figure(go.Bar(
    x=education_counts.values,
    y=education_counts.index,
    orientation='h'
))

# Customize the chart layout
fig.update_layout(
    title='Respondents by Education Category',
    xaxis_title='Number of Respondents',
    yaxis_title='Education Category',
    plot_bgcolor='rgba(0,0,0,0)', # Set transparent background
    title_x=0.5, # Center the title horizontally
    title_y=0.9, # Center the title vertically
    legend=dict(
        orientation='h', # Center the legend horizontally
        yanchor='bottom',
        y=1.02,
        xanchor='center',
        x=0.5
    )
)

# Show the chart
fig.show()


# ## Income

# In[16]:


# Define the income buckets and corresponding labels
income_buckets = [0, 1000, 2000, 3000, float('inf')]
income_labels = ['0-1k', '1k-2k', '2k-3k', '3k+']

# Group the data by income bucket and count the number of responses for each bucket
income_counts = pd.cut(df['Какви са Вашите приблизителни средно месечни доходи?'], bins=income_buckets, labels=income_labels, include_lowest=True).value_counts(sort=False)

# Create a bar chart of the income counts using Plotly's graph objects
fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=income_labels,
        y=income_counts.values,
        marker_color='blue',
        name='Income Counts',
        text=income_counts.values, # Add value labels
        textposition='outside', # Display value labels on top of the bars
    )
)

# Set the plot title and axis labels
fig.update_layout(
    title='Distribution of Respondent Incomes',
    xaxis_title='Income Buckets',
    yaxis_title='Number of Responses',
    plot_bgcolor='rgba(0,0,0,0)', # Set transparent background
    title_x=0.5, # Center the title horizontally
    title_y=0.9, # Center the title vertically
    legend=dict(
        orientation='h', # Horizontal legend
        yanchor='top', # Anchor legend to top of plot
        y=1.1, # Move legend slightly above plot
        xanchor='center', # Center legend horizontally
        x=0.5,
    )
)

# Display the plot
fig.show()


# In[17]:


income_counts


# ## Residence

# In[18]:


df['В какво населено място живеете?'].value_counts()


# In[19]:


df['Location_Clean'] = df['В какво населено място живеете?'].fillna('Other').apply(lambda x: x.split('//')[-1].strip() if '//' in x else 'Other')

# Group the data by living place and count the number of responses for each place
living_counts = df['Location_Clean'].value_counts()
# Replace categories with less than 2 counts with 'Other'
living_counts.loc[living_counts < 2] = 'Other'
# Sort the values in descending order
living_counts = living_counts.sort_values(ascending=False)

# Create a bar chart of the living place counts using Plotly's graph objects
fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=living_counts.index,
        y=living_counts.values,
        marker_color='blue',
        name='Living Place Counts',
        text=living_counts.values, # Add value labels
        textposition='outside', # Display value labels on top of the bars
    )
)

# Set the plot title and axis labels
fig.update_layout(
    title='Distribution of Respondents by Living Place',
    xaxis_title='Living Place',
    yaxis_title='Number of Responses',
    plot_bgcolor='rgba(0,0,0,0)', # Set transparent background
    title_x=0.5, # Center the title horizontally
    title_y=0.9, # Center the title vertically
    legend=dict(orientation='h', yanchor='top', y=-0.1, xanchor='center', x=0.5), # Center the legend horizontally above the graph
)

# Display the plot
fig.show()


# In[20]:


living_counts


# ## Employment

# In[21]:


df['Role'] = df['Каква е Вашата настояща позиция?'].fillna('Other').apply(lambda x: x.split('//')[-1].strip() if '//' in x else 'Other').replace('entrepreneur','Other')


# In[22]:


# Group the data by role and count the number of responses for each role
role_counts = df['Role'].value_counts()

# Create a bar chart of the role counts using Plotly's graph objects
fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=role_counts.index,
        y=role_counts.values,
        marker_color='blue',
        name='Role Counts',
        text=role_counts.values, # Add value labels
        textposition='outside', # Display value labels on top of the bars
    )
)

# Set the plot title and axis labels
fig.update_layout(
    title='Distribution of Respondent Roles',
    xaxis_title='Roles',
    yaxis_title='Number of Responses',
    plot_bgcolor='rgba(0,0,0,0)', # Set transparent background
    title_x=0.5, # Center the title horizontally
    title_y=0.9, # Center the title vertically
)

# Display the plot
fig.show()


# In[23]:


df['Role'].value_counts()[:]


# ## Area

# In[24]:


# Group the values in the "Role" column and map "other" for groupings with counts less than 5
value_counts = df['Каква е Вашата сфера на дейност?'].value_counts()

# Filter to include only groups with 5 or more counts
value_counts_filtered = value_counts[value_counts >= 10]

# Create a new Series with the counts of all other groups
other_counts = pd.Series(value_counts[value_counts < 10].sum(), index=['Other'])

# Concatenate the two Series
final_counts = pd.concat([value_counts_filtered, other_counts])

# Print the final counts
print(final_counts)


# In[25]:


names = ['Trade', 'Education', 'Finance', 'Service','IT','Manifacturing','Transport','Research and Academics','Other']
values = final_counts.values


# In[26]:


names, values


# In[27]:


# Create a bar chart of the industry counts using Plotly's graph objects
fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=names,
        y=values,
        marker_color='blue',
        name='Industry Counts',
        text=values, # Add value labels
        textposition='outside', # Display value labels on top of the bars
    )
)

# Set the plot title and axis labels
fig.update_layout(
    title='Distribution of Respondent Industries',
    xaxis_title='Industry',
    yaxis_title='Number of Responses',
    plot_bgcolor='rgba(0,0,0,0)', # Set transparent background
    title_x=0.5, # Center the title horizontally
    title_y=0.9, # Center the title vertically
)

# Display the plot
fig.show()


# ## You have a choice between two instant deals. 

# In[28]:


q1 = df[[' [X = 1 EU]', ' [X = 10 EU]',
       ' [X = 100 EU]', ' [X = 1 000 EU]', ' [X = 10 000 EU]',
       ' [X = 100 000 EU]']].dropna().astype(int) - 1


# In[29]:


import pandas as pd
import plotly.graph_objects as go

# Assuming your DataFrame 'q1' is already defined

# Get unique rows and their counts
unique_rows = q1.groupby(q1.columns.tolist()).size().reset_index().rename(columns={0: 'count'})

# Exclude rows with counts under 11
unique_rows = unique_rows[unique_rows['count'] >= 11]

# Initialize the node labels, node colors, and source/target connections
labels = ['Start']
colors = ['gray']
source = []
target = []
value = []

# Iterate through the unique rows and build the nodes and connections
for index, row in unique_rows.iterrows():
    current_node = 0
    path = ''
    for column_index, value_in_column in enumerate(row[:-1]):
        column_name = q1.columns[column_index].replace("X =", "").strip()
        risk_status = 'Risked' if value_in_column == 1 else 'Did not risk'
        path += f"{column_name} - {risk_status} / "
        next_node = path

        if next_node not in labels:
            labels.append(next_node)
            colors.append('red' if risk_status == 'Risked' else 'green')

        next_node_index = labels.index(next_node)

        source.append(current_node)
        target.append(next_node_index)
        value.append(row['count'])

        current_node = next_node_index
        
# Calculate the sum count of the nodes that flow into each node
sum_counts = [0] * len(labels)
for s, t, v in zip(source, target, value):
    sum_counts[t] += v
    

    
# Create the node labels with sum count values
node_labels_with_sum_counts = []
for label, color, sum_count in zip(labels, colors, sum_counts):
    count_text = f"{sum_count}" if color != 'gray' else ''
    node_labels_with_sum_counts.append(count_text)


# Create the Sankey diagram with single lines connecting nodes
fig = go.Figure(data=[go.Sankey(
    node=dict(pad=15, thickness=20, line=dict(color='black', width=0.5), label=node_labels_with_sum_counts, color=colors),
    link=dict(source=source, target=target, value=value, color='rgba(0, 0, 0, 0.3)', hovertemplate='%{label}<br>Count: %{value}<extra></extra>'))])

# Set the title and display the Sankey diagram
fig.update_layout(title_text='Sankey Diagram of Risk Choices', font_size=10)

# Add column names above the Sankey diagram
annotations = []
for idx, column_name in enumerate(q1.columns):
    shortened_name = column_name.replace("X =", "").strip()
    annotations.append(
        dict(
            x=(idx+1)/len(q1.columns), y=1.1, xref='paper', yref='paper',
            text=shortened_name, showarrow=False, font=dict(size=12)
        )
    )
    
# Add custom legend for Risked and Not Risked

annotations.append(
    dict(
        x=0.45, y=1.20, xref='paper', yref='paper',
        text='Risked', showarrow=False, font=dict(size=12, color='red')
    )
)

annotations.append(
    dict(
        x=0.55, y=1.20, xref='paper', yref='paper',
        text='Not Risked', showarrow=False, font=dict(size=12, color='green')
    )
)


fig.update_layout(annotations=annotations)

fig.update_layout(
    title_text="",
    font_size=10,
    legend=dict(
        x=0.9,
        y=1,
        traceorder="normal",
        font=dict(size=12),
        bgcolor="rgba(0, 0, 0, 0)",
        bordercolor="rgba(0, 0, 0, 0)",
    ),
)

fig.show()


# In[30]:


import pingouin as pg

# Calculate Krombach's alpha
alpha = pg.cronbach_alpha(q1)

# Print the results
print(alpha)


# In[31]:


q1


# In[32]:


data = q1.copy()


# In[33]:


data = q1.copy()
for col in data.columns:
    # Remove one column from the data
    data_subset = data.drop(columns=col)
    # Calculate Cronbach's alpha for the subset of data
    alpha_subset = pg.cronbach_alpha(data_subset)
    print(f"Cronbach's alpha (without {col}):", alpha_subset)


# In[34]:


import matplotlib.pyplot as plt
import pandas as pd

# Define the alpha coefficients and confidence intervals
alphas = [0.583, 0.560, 0.520, 0.515, 0.598, 0.658]

# Define the column names
cols = ['[1 EU]', '[10 EU]', '[100 EU]', '[1 000 EU]', '[10 000 EU]', '[100 000 EU]']

# Create a pandas DataFrame with the alpha coefficients
df2 = pd.DataFrame({'Removed column': cols, 'Cronbach\'s alpha': alphas})

# Generate the boxplot
ax = df2.boxplot(column='Cronbach\'s alpha', by='Removed column')

# Set the axis labels and title
ax.set_xlabel('Removed column')
ax.set_ylabel('Cronbach\'s alpha')
ax.set_title('Impact of removing each column on Cronbach\'s alpha')

# Show the plot
plt.show()


# In[35]:


data = q1.copy()
data.columns = q1.columns.str.replace('X = ', '')


# In[36]:


corr = data.corr()


# In[37]:


corr.values


# In[38]:


import seaborn as sns

import matplotlib.pyplot as plt
# Create a heatmap of the correlation matrix
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)

# Show the plot
plt.show()


# In[39]:


# Calculate the item-total correlations
total_score = data.sum(axis=1)
corr = []
for col in data.columns:
    col_corr = pg.corr(data[col], total_score, method='pearson').iloc[0, 1]
    corr.append(col_corr)
corr = pd.DataFrame(corr, index=data.columns, columns=['item-total'])


# In[40]:


import plotly.graph_objs as go

# Create the bar trace with text labels above the bars
bar_trace = go.Bar(
    x=corr.index,
    y=corr['item-total'],
    name='Item-Total Correlation',
    text=corr['item-total'].round(2),
    textposition='outside',
    marker=dict(color='blue')
)

# Set the layout
layout = go.Layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(title='Questions'),
    yaxis=dict(title='Item-Total Correlation')
)

# Create the figure and add the trace and layout
fig = go.Figure(data=[bar_trace], layout=layout)

# Show the plot
fig.show()


# ##  You have a choice between two instant deals 2

# In[41]:


q1 = df[[' [X = - 1 EU ]', ' [X = - 10 EU]',
       ' [X = - 100 EU]', ' [X = - 1 000 EU]', ' [X = - 10 000 EU]',
       ' [X = - 100 000 EU]']].dropna().astype(int) - 1


# In[42]:


import pandas as pd
import plotly.graph_objects as go

# Assuming your DataFrame 'q1' is already defined

# Get unique rows and their counts
unique_rows = q1.groupby(q1.columns.tolist()).size().reset_index().rename(columns={0: 'count'})

# Exclude rows with counts under 11
unique_rows = unique_rows[unique_rows['count'] >= 11]

# Initialize the node labels, node colors, and source/target connections
labels = ['Start']
colors = ['gray']
source = []
target = []
value = []

# Iterate through the unique rows and build the nodes and connections
for index, row in unique_rows.iterrows():
    current_node = 0
    path = ''
    for column_index, value_in_column in enumerate(row[:-1]):
        column_name = q1.columns[column_index].replace("X =", "").strip()
        risk_status = 'Risked' if value_in_column == 1 else 'Did not risk'
        path += f"{column_name} - {risk_status} / "
        next_node = path

        if next_node not in labels:
            labels.append(next_node)
            colors.append('red' if risk_status == 'Risked' else 'green')

        next_node_index = labels.index(next_node)

        source.append(current_node)
        target.append(next_node_index)
        value.append(row['count'])

        current_node = next_node_index
        
# Calculate the sum count of the nodes that flow into each node
sum_counts = [0] * len(labels)
for s, t, v in zip(source, target, value):
    sum_counts[t] += v
    

    
# Create the node labels with sum count values
node_labels_with_sum_counts = []
for label, color, sum_count in zip(labels, colors, sum_counts):
    count_text = f"{sum_count}" if color != 'gray' else ''
    node_labels_with_sum_counts.append(count_text)


# Create the Sankey diagram with single lines connecting nodes
fig = go.Figure(data=[go.Sankey(
    node=dict(pad=15, thickness=20, line=dict(color='black', width=0.5), label=node_labels_with_sum_counts, color=colors),
    link=dict(source=source, target=target, value=value, color='rgba(0, 0, 0, 0.3)', hovertemplate='%{label}<br>Count: %{value}<extra></extra>'))])

# Set the title and display the Sankey diagram
fig.update_layout(title_text='Sankey Diagram of Risk Choices', font_size=10)

# Add column names above the Sankey diagram
annotations = []
for idx, column_name in enumerate(q1.columns):
    shortened_name = column_name.replace("X =", "").strip()
    annotations.append(
        dict(
            x=(idx+1)/len(q1.columns), y=1.1, xref='paper', yref='paper',
            text=shortened_name, showarrow=False, font=dict(size=12)
        )
    )
    
# Add custom legend for Risked and Not Risked

annotations.append(
    dict(
        x=0.45, y=1.20, xref='paper', yref='paper',
        text='Risked', showarrow=False, font=dict(size=12, color='red')
    )
)

annotations.append(
    dict(
        x=0.55, y=1.20, xref='paper', yref='paper',
        text='Not Risked', showarrow=False, font=dict(size=12, color='green')
    )
)


fig.update_layout(annotations=annotations)

fig.update_layout(
    title_text="",
    font_size=10,
    legend=dict(
        x=0.9,
        y=1,
        traceorder="normal",
        font=dict(size=12),
        bgcolor="rgba(0, 0, 0, 0)",
        bordercolor="rgba(0, 0, 0, 0)",
    ),
)

fig.show()


# In[43]:


# Calculate Krombach's alpha
alpha = pg.cronbach_alpha(q1)

# Print the results
print(alpha)


# In[44]:


data = q1.copy()
for col in data.columns:
    # Remove one column from the data
    data_subset = data.drop(columns=col)
    # Calculate Cronbach's alpha for the subset of data
    alpha_subset = pg.cronbach_alpha(data_subset)
    print(f"Cronbach's alpha (without {col}):", alpha_subset)


# In[45]:


# Calculate the item-total correlations
total_score = data.sum(axis=1)
corr = []
for col in data.columns:
    col_corr = pg.corr(data[col], total_score, method='pearson').iloc[0, 1]
    corr.append(col_corr)
corr = pd.DataFrame(corr, index=data.columns, columns=['item-total'])


# In[46]:


import plotly.graph_objs as go

# Create the bar trace with text labels above the bars
bar_trace = go.Bar(
    x=corr.index,
    y=corr['item-total'],
    name='Item-Total Correlation',
    text=corr['item-total'].round(2),
    textposition='outside',
    marker=dict(color='blue')
)

# Set the layout
layout = go.Layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(title='Questions'),
    yaxis=dict(title='Item-Total Correlation')
)

# Create the figure and add the trace and layout
fig = go.Figure(data=[bar_trace], layout=layout)

# Show the plot
fig.show()


# ## First 12

# In[47]:


df.columns


# In[48]:


f_12 = df[[' [X = 1 EU]', ' [X = 10 EU]',
       ' [X = 100 EU]', ' [X = 1 000 EU]', ' [X = 10 000 EU]',
       ' [X = 100 000 EU]', ' [X = - 1 EU ]', ' [X = - 10 EU]',
       ' [X = - 100 EU]', ' [X = - 1 000 EU]', ' [X = - 10 000 EU]',
       ' [X = - 100 000 EU]']].dropna()


# In[49]:


f_12


# In[50]:


# Calculate Krombach's alpha
alpha = pg.cronbach_alpha(f_12)

# Print the results
print(alpha)


# In[51]:


data = f_12.copy()
for col in data.columns:
    # Remove one column from the data
    data_subset = f_12.drop(columns=col)
    # Calculate Cronbach's alpha for the subset of data
    alpha_subset = pg.cronbach_alpha(data_subset)
    print(f"Cronbach's alpha (without {col}):", alpha_subset)


# In[52]:


# Calculate the item-total correlations
total_score = f_12.sum(axis=1)
corr = []
for col in f_12.columns:
    col_corr = pg.corr(f_12[col], total_score, method='pearson').iloc[0, 1]
    corr.append(col_corr)
corr = pd.DataFrame(corr, index=f_12.columns, columns=['item-total'])

import plotly.graph_objs as go

# Create the bar trace with text labels above the bars
bar_trace = go.Bar(
    x=corr.index,
    y=corr['item-total'],
    name='Item-Total Correlation',
    text=corr['item-total'].round(2),
    textposition='outside',
    marker=dict(color='blue')
)

# Set the layout
layout = go.Layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(title='Questions'),
    yaxis=dict(title='Item-Total Correlation')
)

# Create the figure and add the trace and layout
fig = go.Figure(data=[bar_trace], layout=layout)

# Show the plot
fig.show()


# ## 3rd 3

# In[53]:


df.columns


# In[54]:


f_18 = df[['Y=2 EU', 'Y=20 EU', 'Y=200 EU', 'Y=2 000 EU',
       'Y=20 000 EU', 'Y=200 000 EU']].dropna()


# In[55]:


f_18


# In[56]:


# f_18['2 EUR'] = np.where(f_18['Y=2 EU'] <= 1, 0, 1)
# f_18['20 EUR'] = np.where(f_18['Y=20 EU'] <= 10, 0, 1)
# f_18['200 EUR'] =  np.where(f_18['Y=200 EU'] <= 100, 0, 1)
# f_18['2000 EUR'] = np.where(f_18['Y=2 000 EU'] <= 1000, 0, 1)
# f_18['20 000 EUR'] =  np.where(f_18['Y=20 000 EU'] <= 10000, 0, 1)
# f_18['200 000 EUR'] = np.where(f_18['Y=200 000 EU'] <= 100000, 0, 1)


# In[57]:


f_18['2 EUR'] = f_18['Y=2 EU'].apply(lambda x: 0 if x == 1 else (-1 if x < 1 else 1))
f_18['20 EUR'] = (f_18['Y=20 EU']/10).apply(lambda x: 0 if x == 1 else (-1 if x < 1 else 1))
f_18['200 EUR'] =  (f_18['Y=200 EU']/100).apply(lambda x: 0 if x == 1 else (-1 if x < 1 else 1))
f_18['2000 EUR'] = (f_18['Y=2 000 EU']/1000).apply(lambda x: 0 if x == 1 else (-1 if x < 1 else 1))
f_18['20 000 EUR'] = (f_18['Y=20 000 EU']/10000).apply(lambda x: 0 if x == 1 else (-1 if x < 1 else 1))
f_18['200 000 EUR'] = (f_18['Y=200 000 EU']/100000).apply(lambda x: 0 if x == 1 else (-1 if x < 1 else 1))


# In[58]:


q1 = f_18[['2 EUR', '20 EUR', '200 EUR', '2000 EUR','20 000 EUR', '200 000 EUR' ]]


# In[59]:


q18 = f_18[['2 EUR', '20 EUR', '200 EUR', '2000 EUR','20 000 EUR', '200 000 EUR' ]]


# In[60]:


import pandas as pd
import plotly.graph_objects as go

# Assuming your DataFrame 'q1' is already defined

# Get unique rows and their counts
unique_rows = q1.groupby(q1.columns.tolist()).size().reset_index().rename(columns={0: 'count'})

# Exclude rows with counts under 11
unique_rows = unique_rows[unique_rows['count'] >= 15]

# Initialize the node labels, node colors, and source/target connections
labels = ['Start']
colors = ['gray']
source = []
target = []
value = []

# Iterate through the unique rows and build the nodes and connections
for index, row in unique_rows.iterrows():
    current_node = 0
    path = ''
    for column_index, value_in_column in enumerate(row[:-1]):
        column_name = q1.columns[column_index].replace("X =", "").strip()
        risk_status = 'Premium' if value_in_column == 1 else 'Discount'
        path += f"{column_name} - {risk_status} / "
        next_node = path

        if next_node not in labels:
            labels.append(next_node)
            colors.append('red' if risk_status == 'Premium' else 'green')

        next_node_index = labels.index(next_node)

        source.append(current_node)
        target.append(next_node_index)
        value.append(row['count'])

        current_node = next_node_index
        
# Calculate the sum count of the nodes that flow into each node
sum_counts = [0] * len(labels)
for s, t, v in zip(source, target, value):
    sum_counts[t] += v
    

    
# Create the node labels with sum count values
node_labels_with_sum_counts = []
for label, color, sum_count in zip(labels, colors, sum_counts):
    count_text = f"{sum_count}" if color != 'gray' else ''
    node_labels_with_sum_counts.append(count_text)


# Create the Sankey diagram with single lines connecting nodes
fig = go.Figure(data=[go.Sankey(
    node=dict(pad=15, thickness=20, line=dict(color='black', width=0.5), label=node_labels_with_sum_counts, color=colors),
    link=dict(source=source, target=target, value=value, color='rgba(0, 0, 0, 0.3)', hovertemplate='%{label}<br>Count: %{value}<extra></extra>'))])

# Set the title and display the Sankey diagram
fig.update_layout(title_text='Sankey Diagram of Risk Choices', font_size=10)

# Add column names above the Sankey diagram
annotations = []
for idx, column_name in enumerate(q1.columns):
    shortened_name = column_name.replace("X =", "").strip()
    annotations.append(
        dict(
            x=(idx+1)/len(q1.columns), y=1.1, xref='paper', yref='paper',
            text=shortened_name, showarrow=False, font=dict(size=12)
        )
    )
    
# Add custom legend for Risked and Not Risked

annotations.append(
    dict(
        x=0.45, y=1.20, xref='paper', yref='paper',
        text='Premium', showarrow=False, font=dict(size=12, color='red')
    )
)

annotations.append(
    dict(
        x=0.55, y=1.20, xref='paper', yref='paper',
        text='Discount', showarrow=False, font=dict(size=12, color='green')
    )
)


fig.update_layout(annotations=annotations)

fig.update_layout(
    title_text="",
    font_size=10,
    legend=dict(
        x=0.9,
        y=1,
        traceorder="normal",
        font=dict(size=12),
        bgcolor="rgba(0, 0, 0, 0)",
        bordercolor="rgba(0, 0, 0, 0)",
    ),
)

fig.show()


# In[61]:


q1


# In[62]:


import pandas as pd
import plotly.graph_objects as go

# Assuming your DataFrame 'q1' is already defined

# Get unique rows and their counts
unique_rows = q1.groupby(q1.columns.tolist()).size().reset_index().rename(columns={0: 'count'})

# Exclude rows with counts under 11
unique_rows = unique_rows[unique_rows['count'] >= 17]

# Initialize the node labels, node colors, and source/target connections
labels = ['Start']
colors = ['gray']
source = []
target = []
value = []

# Iterate through the unique rows and build the nodes and connections
for index, row in unique_rows.iterrows():
    current_node = 0
    path = ''
    for column_index, value_in_column in enumerate(row[:-1]):
        column_name = q1.columns[column_index].replace("X =", "").strip()
        if value_in_column == 1:
            risk_status = 'Premium'
        elif value_in_column == -1:
            risk_status = 'Discount'
        elif value_in_column == 0:
            risk_status = 'Neutral'
        path += f"{column_name} - {risk_status} / "
        next_node = path

        if next_node not in labels:
            labels.append(next_node)
            if risk_status == 'Premium':
                colors.append('red')
            elif risk_status == 'Discount':
                colors.append('green')
            else:
                colors.append('blue')

        next_node_index = labels.index(next_node)

        source.append(current_node)
        target.append(next_node_index)
        value.append(row['count'])

        current_node = next_node_index
        
# Calculate the sum count of the nodes that flow into each node
sum_counts = [0] * len(labels)
for s, t, v in zip(source, target, value):
    sum_counts[t] += v
    

# Create the node labels with sum count values
node_labels_with_sum_counts = []
for label, color, sum_count in zip(labels, colors, sum_counts):
    count_text = f"{sum_count}" if color != 'gray' else ''
    node_labels_with_sum_counts.append(count_text)

# Create the Sankey diagram with single lines connecting nodes
fig = go.Figure(data=[go.Sankey(
    node=dict(pad=15, thickness=20, line=dict(color='black', width=0.5), label=node_labels_with_sum_counts, color=colors),
    link=dict(source=source, target=target, value=value, color='rgba(0, 0, 0, 0.3)', hovertemplate='%{label}<br>Count: %{value}<extra></extra>'))])

# Set the title and display the Sankey diagram
fig.update_layout(title_text='Sankey Diagram of Risk Choices', font_size=10)

# Add column names above the Sankey diagram
annotations = []
for idx, column_name in enumerate(q1.columns):
    shortened_name = column_name.replace("X =", "").strip()
    annotations.append(
        dict(
            x=(idx+1)/len(q1.columns), y=1.1, xref='paper', yref='paper',
            text=shortened_name, showarrow=False, font=dict(size=12)
        )
    )
    

# Add custom legend for Risked and Not Risked

annotations.append(
    dict(
        x=0.45, y=1.20, xref='paper', yref='paper',
        text='Premium', showarrow=False, font=dict(size=12, color='red')
    )
)

annotations.append(
    dict(
        x=0.55, y=1.20, xref='paper', yref='paper',
        text='Discount', showarrow=False, font=dict(size=12, color='green')
    )
)

annotations.append(
    dict(
        x=0.65, y=1.20, xref='paper', yref='paper',
        text='Neutral', showarrow=False, font=dict(size=12, color='blue')
    )
)

# ...
fig.update_layout(annotations=annotations)

fig.update_layout(
    title_text="",
    font_size=10,
    legend=dict(
        x=0.9,
        y=1,
        traceorder="normal",
        font=dict(size=12),
        bgcolor="rgba(0, 0, 0, 0)",
        bordercolor="rgba(0, 0, 0, 0)",
    ),
)

fig.show()


# In[63]:


# Calculate Krombach's alpha
alpha = pg.cronbach_alpha(df[['Y=2 EU', 'Y=20 EU', 'Y=200 EU', 'Y=2 000 EU',
       'Y=20 000 EU', 'Y=200 000 EU']].dropna())

# Print the results
print(alpha)


# In[64]:


# Calculate Krombach's alpha
alpha = pg.cronbach_alpha(q1)

# Print the results
print(alpha)


# In[65]:


data = df[['Y=2 EU', 'Y=20 EU', 'Y=200 EU', 'Y=2 000 EU',
       'Y=20 000 EU', 'Y=200 000 EU']].dropna().copy()
for col in data.columns:
    # Remove one column from the data
    data_subset = f_18.drop(columns=col)
    # Calculate Cronbach's alpha for the subset of data
    alpha_subset = pg.cronbach_alpha(data_subset)
    print(f"Cronbach's alpha (without {col}):", alpha_subset)


# In[66]:


corr


# In[67]:


# Calculate the item-total correlations
total_score = data.sum(axis=1)
corr = []
for col in data.columns:
    col_corr = pg.corr(data[col], total_score, method='pearson').iloc[0, 1]
    corr.append(col_corr)
corr = pd.DataFrame(corr, index=data.columns, columns=['item-total'])

import plotly.graph_objs as go

# Create the bar trace with text labels above the bars
bar_trace = go.Bar(
    x=corr.index,
    y=corr['item-total'],
    name='Item-Total Correlation',
    text=corr['item-total'].round(2),
    textposition='outside',
    marker=dict(color='blue')
)

# Set the layout
layout = go.Layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(title='Questions'),
    yaxis=dict(title='Item-Total Correlation')
)

# Create the figure and add the trace and layout
fig = go.Figure(data=[bar_trace], layout=layout)

# Show the plot
fig.show()


# In[68]:


df[[' [X = 1 EU]', ' [X = 10 EU]',
       ' [X = 100 EU]', ' [X = 1 000 EU]', ' [X = 10 000 EU]',
       ' [X = 100 000 EU]', ' [X = - 1 EU ]', ' [X = - 10 EU]',
       ' [X = - 100 EU]', ' [X = - 1 000 EU]', ' [X = - 10 000 EU]',
       ' [X = - 100 000 EU]', 'Y=2 EU', 'Y=20 EU', 'Y=200 EU', 'Y=2 000 EU',
       'Y=20 000 EU', 'Y=200 000 EU']].dropna()


# In[ ]:





# In[69]:


# Calculate the item-total correlations
total_score = df[[' [X = 1 EU]', ' [X = 10 EU]',
       ' [X = 100 EU]', ' [X = 1 000 EU]', ' [X = 10 000 EU]',
       ' [X = 100 000 EU]', ' [X = - 1 EU ]', ' [X = - 10 EU]',
       ' [X = - 100 EU]', ' [X = - 1 000 EU]', ' [X = - 10 000 EU]',
       ' [X = - 100 000 EU]', 'Y=2 EU', 'Y=20 EU', 'Y=200 EU', 'Y=2 000 EU',
       'Y=20 000 EU', 'Y=200 000 EU']].dropna().sum(axis=1)
corr = []
for col in df[[' [X = 1 EU]', ' [X = 10 EU]',
       ' [X = 100 EU]', ' [X = 1 000 EU]', ' [X = 10 000 EU]',
       ' [X = 100 000 EU]', ' [X = - 1 EU ]', ' [X = - 10 EU]',
       ' [X = - 100 EU]', ' [X = - 1 000 EU]', ' [X = - 10 000 EU]',
       ' [X = - 100 000 EU]', 'Y=2 EU', 'Y=20 EU', 'Y=200 EU', 'Y=2 000 EU',
       'Y=20 000 EU', 'Y=200 000 EU']].dropna().columns:
    col_corr = pg.corr(df[[' [X = 1 EU]', ' [X = 10 EU]',
       ' [X = 100 EU]', ' [X = 1 000 EU]', ' [X = 10 000 EU]',
       ' [X = 100 000 EU]', ' [X = - 1 EU ]', ' [X = - 10 EU]',
       ' [X = - 100 EU]', ' [X = - 1 000 EU]', ' [X = - 10 000 EU]',
       ' [X = - 100 000 EU]', 'Y=2 EU', 'Y=20 EU', 'Y=200 EU', 'Y=2 000 EU',
       'Y=20 000 EU', 'Y=200 000 EU']].dropna()[col], total_score, method='pearson').iloc[0, 1]
    corr.append(col_corr)
corr = pd.DataFrame(corr, index=df[[' [X = 1 EU]', ' [X = 10 EU]',
       ' [X = 100 EU]', ' [X = 1 000 EU]', ' [X = 10 000 EU]',
       ' [X = 100 000 EU]', ' [X = - 1 EU ]', ' [X = - 10 EU]',
       ' [X = - 100 EU]', ' [X = - 1 000 EU]', ' [X = - 10 000 EU]',
       ' [X = - 100 000 EU]', 'Y=2 EU', 'Y=20 EU', 'Y=200 EU', 'Y=2 000 EU',
       'Y=20 000 EU', 'Y=200 000 EU']].dropna().columns, columns=['item-total'])

import plotly.graph_objs as go

# Create the bar trace with text labels above the bars
bar_trace = go.Bar(
    x=corr.index,
    y=corr['item-total'],
    name='Item-Total Correlation',
    text=corr['item-total'].round(2),
    textposition='outside',
    marker=dict(color='blue')
)

# Set the layout
layout = go.Layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(title='Questions'),
    yaxis=dict(title='Item-Total Correlation')
)

# Create the figure and add the trace and layout
fig = go.Figure(data=[bar_trace], layout=layout)

# Show the plot
fig.show()


# In[70]:


data = df[['Y=2 EU', 'Y=20 EU', 'Y=200 EU', 'Y=2 000 EU',
       'Y=20 000 EU', 'Y=200 000 EU']].dropna().copy()
for col in q1.columns:
    # Remove one column from the data
    data_subset = q1.drop(columns=col)
    # Calculate Cronbach's alpha for the subset of data
    alpha_subset = pg.cronbach_alpha(data_subset)
    print(f"Cronbach's alpha (without {col}):", alpha_subset)


# In[71]:


q1[['2 EUR', '20 EUR', '200 EUR', '2000 EUR', '20 000 EUR', '200 000 EUR']]


# In[72]:


df[[' [X = 1 EU]', ' [X = 10 EU]',
       ' [X = 100 EU]', ' [X = 1 000 EU]', ' [X = 10 000 EU]',
       ' [X = 100 000 EU]', ' [X = - 1 EU ]', ' [X = - 10 EU]',
       ' [X = - 100 EU]', ' [X = - 1 000 EU]', ' [X = - 10 000 EU]',
       ' [X = - 100 000 EU]']].dropna().append(q1[['2 EUR', '20 EUR', '200 EUR', '2000 EUR', '20 000 EUR', '200 000 EUR']])


# In[73]:


df_combined = pd.concat([df[[' [X = 1 EU]', ' [X = 10 EU]', ' [X = 100 EU]', ' [X = 1 000 EU]', ' [X = 10 000 EU]', ' [X = 100 000 EU]', ' [X = - 1 EU ]', ' [X = - 10 EU]', ' [X = - 100 EU]', ' [X = - 1 000 EU]', ' [X = - 10 000 EU]', ' [X = - 100 000 EU]']].dropna(), q1[['2 EUR', '20 EUR', '200 EUR', '2000 EUR', '20 000 EUR', '200 000 EUR']]], axis=1)
df_combined


# In[74]:


pg.cronbach_alpha(df_combined)


# In[75]:


for col in df_combined.columns:
    # Remove one column from the data
    data_subset = df_combined.drop(columns=col)
    # Calculate Cronbach's alpha for the subset of data
    alpha_subset = pg.cronbach_alpha(data_subset)
    print(f"Cronbach's alpha (without {col}):", alpha_subset)


# In[76]:


# Calculate the item-total correlations
total_score = df_combined.sum(axis=1)
corr = []
for col in df_combined.columns:
    col_corr = pg.corr(df_combined[col], total_score, method='pearson').iloc[0, 1]
    corr.append(col_corr)
corr = pd.DataFrame(corr, index=df_combined.columns, columns=['item-total'])

import plotly.graph_objs as go

# Create the bar trace with text labels above the bars
bar_trace = go.Bar(
    x=corr.index,
    y=corr['item-total'],
    name='Item-Total Correlation',
    text=corr['item-total'].round(2),
    textposition='outside',
    marker=dict(color='blue')
)

# Set the layout
layout = go.Layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(title='Questions'),
    yaxis=dict(title='Item-Total Correlation')
)

# Create the figure and add the trace and layout
fig = go.Figure(data=[bar_trace], layout=layout)

# Show the plot
fig.show()


# In[77]:


corr


# In[78]:


df.columns


# ## Last set of questions

# In[79]:


f_24 = df[['Y= - 2 EU', 'Y= - 20 EU', 'Y= - 200 EU',
       'Y= - 2 000 EU', 'Y= - 20 000 EU', 'Y= - 200 000 EU']].dropna()


# In[80]:


f_24[:30]


# In[81]:


f_24.describe().to_csv('esketit.csv')


# In[82]:


f_24['-2 EUR'] = (f_24['Y= - 2 EU']/-1).apply(lambda x: 0 if x == 1 else (-1 if x < 1 else 1))
f_24['-20 EUR'] = (f_24['Y= - 20 EU']/-10).apply(lambda x: 0 if x == 1 else (-1 if x < 1 else 1))
f_24['-200 EUR'] =  (f_24['Y= - 200 EU']/-100).apply(lambda x: 0 if x == 1 else (-1 if x < 1 else 1))
f_24['-2000 EUR'] = (f_24['Y= - 2 000 EU']/-1000).apply(lambda x: 0 if x == 1 else (-1 if x < 1 else 1))
f_24['-20 000 EUR'] = (f_24['Y= - 20 000 EU']/-10000).apply(lambda x: 0 if x == 1 else (-1 if x < 1 else 1))
f_24['-200 000 EUR'] = (f_24['Y= - 200 000 EU']/-100000).apply(lambda x: 0 if x == 1 else (-1 if x < 1 else 1))


# In[83]:


f_24


# In[84]:


q1 = f_24[['-2 EUR', '-20 EUR', '-200 EUR',
       '-2000 EUR', '-20 000 EUR', '-200 000 EUR']] 

q24 = f_24[['-2 EUR', '-20 EUR', '-200 EUR',
       '-2000 EUR', '-20 000 EUR', '-200 000 EUR']] 


# In[85]:


q1[:30]


# In[86]:


import pandas as pd
import plotly.graph_objects as go

# Assuming your DataFrame 'q1' is already defined

# Get unique rows and their counts
unique_rows = q1.groupby(q1.columns.tolist()).size().reset_index().rename(columns={0: 'count'})

# Exclude rows with counts under 11
unique_rows = unique_rows[unique_rows['count'] >= 15]

# Initialize the node labels, node colors, and source/target connections
labels = ['Start']
colors = ['gray']
source = []
target = []
value = []

# Iterate through the unique rows and build the nodes and connections
for index, row in unique_rows.iterrows():
    current_node = 0
    path = ''
    for column_index, value_in_column in enumerate(row[:-1]):
        column_name = q1.columns[column_index].replace("X =", "").strip()
        if value_in_column == 1:
            risk_status = 'Premium'
        elif value_in_column == -1:
            risk_status = 'Discount'
        elif value_in_column == 0:
            risk_status = 'Neutral'
        path += f"{column_name} - {risk_status} / "
        next_node = path

        if next_node not in labels:
            labels.append(next_node)
            if risk_status == 'Premium':
                colors.append('red')
            elif risk_status == 'Discount':
                colors.append('green')
            else:
                colors.append('blue')

        next_node_index = labels.index(next_node)

        source.append(current_node)
        target.append(next_node_index)
        value.append(row['count'])

        current_node = next_node_index
        
# Calculate the sum count of the nodes that flow into each node
sum_counts = [0] * len(labels)
for s, t, v in zip(source, target, value):
    sum_counts[t] += v
    

# Create the node labels with sum count values
node_labels_with_sum_counts = []
for label, color, sum_count in zip(labels, colors, sum_counts):
    count_text = f"{sum_count}" if color != 'gray' else ''
    node_labels_with_sum_counts.append(count_text)

# Create the Sankey diagram with single lines connecting nodes
fig = go.Figure(data=[go.Sankey(
    node=dict(pad=15, thickness=20, line=dict(color='black', width=0.5), label=node_labels_with_sum_counts, color=colors),
    link=dict(source=source, target=target, value=value, color='rgba(0, 0, 0, 0.3)', hovertemplate='%{label}<br>Count: %{value}<extra></extra>'))])

# Set the title and display the Sankey diagram
fig.update_layout(title_text='Sankey Diagram of Risk Choices', font_size=10)

# Add column names above the Sankey diagram
annotations = []
for idx, column_name in enumerate(q1.columns):
    shortened_name = column_name.replace("X =", "").strip()
    annotations.append(
        dict(
            x=(idx+1)/len(q1.columns), y=1.1, xref='paper', yref='paper',
            text=shortened_name, showarrow=False, font=dict(size=12)
        )
    )
    

# Add custom legend for Risked and Not Risked

annotations.append(
    dict(
        x=0.45, y=1.20, xref='paper', yref='paper',
        text='Premium', showarrow=False, font=dict(size=12, color='red')
    )
)

annotations.append(
    dict(
        x=0.55, y=1.20, xref='paper', yref='paper',
        text='Discount', showarrow=False, font=dict(size=12, color='green')
    )
)

annotations.append(
    dict(
        x=0.65, y=1.20, xref='paper', yref='paper',
        text='Neutral', showarrow=False, font=dict(size=12, color='blue')
    )
)

# ...
fig.update_layout(annotations=annotations)

fig.update_layout(
    title_text="",
    font_size=10,
    legend=dict(
        x=0.9,
        y=1,
        traceorder="normal",
        font=dict(size=12),
        bgcolor="rgba(0, 0, 0, 0)",
        bordercolor="rgba(0, 0, 0, 0)",
    ),
)

fig.show()


# In[ ]:





# In[87]:


pg.cronbach_alpha(df[['Y= - 2 EU', 'Y= - 20 EU', 'Y= - 200 EU',
       'Y= - 2 000 EU', 'Y= - 20 000 EU', 'Y= - 200 000 EU']].dropna())


# In[88]:


pg.cronbach_alpha(q1.dropna())


# In[89]:


df_combined = q1


# In[90]:


# Calculate the item-total correlations
total_score = df_combined.sum(axis=1)
corr = []
for col in df_combined.columns:
    col_corr = pg.corr(df_combined[col], total_score, method='pearson').iloc[0, 1]
    corr.append(col_corr)
corr = pd.DataFrame(corr, index=df_combined.columns, columns=['item-total'])

import plotly.graph_objs as go

# Create the bar trace with text labels above the bars
bar_trace = go.Bar(
    x=corr.index,
    y=corr['item-total'],
    name='Item-Total Correlation',
    text=corr['item-total'].round(2),
    textposition='outside',
    marker=dict(color='blue')
)

# Set the layout
layout = go.Layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(title='Questions'),
    yaxis=dict(title='Item-Total Correlation')
)

# Create the figure and add the trace and layout
fig = go.Figure(data=[bar_trace], layout=layout)

# Show the plot
fig.show()


# In[91]:


pg.cronbach_alpha(df[['Y=2 EU', 'Y=20 EU', 'Y=200 EU', 'Y=2 000 EU',
       'Y=20 000 EU', 'Y=200 000 EU', 'Y= - 2 EU', 'Y= - 20 EU', 'Y= - 200 EU',
       'Y= - 2 000 EU', 'Y= - 20 000 EU', 'Y= - 200 000 EU']].dropna())


# In[92]:


pg.cronbach_alpha(abs(df[['Y=2 EU', 'Y=20 EU', 'Y=200 EU', 'Y=2 000 EU',
       'Y=20 000 EU', 'Y=200 000 EU', 'Y= - 2 EU', 'Y= - 20 EU', 'Y= - 200 EU',
       'Y= - 2 000 EU', 'Y= - 20 000 EU', 'Y= - 200 000 EU']].dropna()))


# In[93]:


f_18.columns


# In[94]:


pg.cronbach_alpha(pd.concat([q18,q24], axis =1))


# In[95]:


pg.cronbach_alpha(pd.concat([df[[' [X = 1 EU]', ' [X = 10 EU]',
       ' [X = 100 EU]', ' [X = 1 000 EU]', ' [X = 10 000 EU]',
       ' [X = 100 000 EU]', ' [X = - 1 EU ]', ' [X = - 10 EU]',
       ' [X = - 100 EU]', ' [X = - 1 000 EU]', ' [X = - 10 000 EU]',
       ' [X = - 100 000 EU]']].dropna(),q18,q24], axis =1))


# In[96]:


df.columns


# In[97]:


pg.cronbach_alpha(df[[' [X = 1 EU]', ' [X = 10 EU]',
       ' [X = 100 EU]', ' [X = 1 000 EU]', ' [X = 10 000 EU]',
       ' [X = 100 000 EU]', ' [X = - 1 EU ]', ' [X = - 10 EU]',
       ' [X = - 100 EU]', ' [X = - 1 000 EU]', ' [X = - 10 000 EU]',
       ' [X = - 100 000 EU]', 'Y=2 EU', 'Y=20 EU', 'Y=200 EU', 'Y=2 000 EU',
       'Y=20 000 EU', 'Y=200 000 EU', 'Y= - 2 EU', 'Y= - 20 EU', 'Y= - 200 EU',
       'Y= - 2 000 EU', 'Y= - 20 000 EU', 'Y= - 200 000 EU']].dropna())


# In[98]:


df_combined = pd.concat([df[[' [X = 1 EU]', ' [X = 10 EU]',
       ' [X = 100 EU]', ' [X = 1 000 EU]', ' [X = 10 000 EU]',
       ' [X = 100 000 EU]', ' [X = - 1 EU ]', ' [X = - 10 EU]',
       ' [X = - 100 EU]', ' [X = - 1 000 EU]', ' [X = - 10 000 EU]',
       ' [X = - 100 000 EU]']].dropna(),q18,q24], axis =1)

# Calculate the item-total correlations
total_score = df_combined.sum(axis=1)
corr = []
for col in df_combined.columns:
    col_corr = pg.corr(df_combined[col], total_score, method='pearson').iloc[0, 1]
    corr.append(col_corr)
corr = pd.DataFrame(corr, index=df_combined.columns, columns=['item-total'])

import plotly.graph_objs as go

# Create the bar trace with text labels above the bars
bar_trace = go.Bar(
    x=corr.index,
    y=corr['item-total'],
    name='Item-Total Correlation',
    text=corr['item-total'].round(2),
    textposition='outside',
    marker=dict(color='blue')
)

# Set the layout
layout = go.Layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(title='Questions'),
    yaxis=dict(title='Item-Total Correlation')
)

# Create the figure and add the trace and layout
fig = go.Figure(data=[bar_trace], layout=layout)

# Show the plot
fig.show()


# In[99]:


df_combined = pd.concat([df[[' [X = 1 EU]', ' [X = 10 EU]',
       ' [X = 100 EU]', ' [X = 1 000 EU]', ' [X = 10 000 EU]',
       ' [X = 100 000 EU]', ' [X = - 1 EU ]', ' [X = - 10 EU]',
       ' [X = - 100 EU]', ' [X = - 1 000 EU]', ' [X = - 10 000 EU]',
       ' [X = - 100 000 EU]']].dropna(),q18,q24], axis =1)

# Calculate the item-total correlations
total_score = df_combined.sum(axis=1)
corr = []
for col in df_combined.columns:
    col_corr = pg.corr(df_combined[col], total_score, method='pearson').iloc[0, 1]
    corr.append(col_corr)
corr = pd.DataFrame(corr, index=df_combined.columns, columns=['item-total'])

import plotly.graph_objs as go

# Create the bar trace with text labels above the bars
bar_trace = go.Bar(
    x=corr.index,
    y=corr['item-total'],
    name='Item-Total Correlation',
    text=corr['item-total'].round(2),
    textposition='outside',
    marker=dict(color='blue')
)

# Set the layout
layout = go.Layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(title='Questions'),
    yaxis=dict(title='Item-Total Correlation')
)

# Create the figure and add the trace and layout
fig = go.Figure(data=[bar_trace], layout=layout)

# Show the plot
fig.show()


# In[ ]:





# In[100]:


df_combined = df[[' [X = 1 EU]', ' [X = 10 EU]',
       ' [X = 100 EU]', ' [X = 1 000 EU]', ' [X = 10 000 EU]',
       ' [X = 100 000 EU]', ' [X = - 1 EU ]', ' [X = - 10 EU]',
       ' [X = - 100 EU]', ' [X = - 1 000 EU]', ' [X = - 10 000 EU]',
       ' [X = - 100 000 EU]', 'Y=2 EU', 'Y=20 EU', 'Y=200 EU', 'Y=2 000 EU',
       'Y=20 000 EU', 'Y=200 000 EU', 'Y= - 2 EU', 'Y= - 20 EU', 'Y= - 200 EU',
       'Y= - 2 000 EU', 'Y= - 20 000 EU', 'Y= - 200 000 EU']]

# Calculate the item-total correlations
total_score = df_combined.sum(axis=1)
corr = []
for col in df_combined.columns:
    col_corr = pg.corr(df_combined[col], total_score, method='pearson').iloc[0, 1]
    corr.append(col_corr)
corr = pd.DataFrame(corr, index=df_combined.columns, columns=['item-total'])

import plotly.graph_objs as go

# Create the bar trace with text labels above the bars
bar_trace = go.Bar(
    x=corr.index,
    y=corr['item-total'],
    name='Item-Total Correlation',
    text=corr['item-total'].round(2),
    textposition='outside',
    marker=dict(color='blue')
)

# Set the layout
layout = go.Layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(title='Questions'),
    yaxis=dict(title='Item-Total Correlation')
)

# Create the figure and add the trace and layout
fig = go.Figure(data=[bar_trace], layout=layout)

# Show the plot
fig.show()


# In[ ]:





# ## General stuff

# In[101]:


df.columns


# In[ ]:





# In[102]:


inter_item_correlations =pd.concat([df[[' [X = 1 EU]', ' [X = 10 EU]',
       ' [X = 100 EU]', ' [X = 1 000 EU]', ' [X = 10 000 EU]',
       ' [X = 100 000 EU]', ' [X = - 1 EU ]', ' [X = - 10 EU]',
       ' [X = - 100 EU]', ' [X = - 1 000 EU]', ' [X = - 10 000 EU]',
       ' [X = - 100 000 EU]']].dropna(),q18,q24], axis =1).corr()


# In[103]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# # Create a pivot table to reshape the data
# corr_matrix = results_df.pivot(index='Sets', columns='Sets', values='Alpha')

# Create a correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(inter_item_correlations, annot=False, cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.xlabel('Sets')
plt.ylabel('Sets')
plt.show()


# In[104]:


work_around = pd.concat([df[[' [X = 1 EU]', ' [X = 10 EU]',
       ' [X = 100 EU]', ' [X = 1 000 EU]', ' [X = 10 000 EU]',
       ' [X = 100 000 EU]', ' [X = - 1 EU ]', ' [X = - 10 EU]',
       ' [X = - 100 EU]', ' [X = - 1 000 EU]', ' [X = - 10 000 EU]',
       ' [X = - 100 000 EU]']].dropna(),q18,q24], axis = 1)


# In[105]:


import pandas as pd

# Subset the DataFrame to include only the columns of interest and drop rows with missing values
# work_around = df[[' [X = 1 EU]', ' [X = 10 EU]', ' [X = 100 EU]', ' [X = 1 000 EU]', ' [X = 10 000 EU]', ' [X = 100 000 EU]', ' [X = - 1 EU ]', ' [X = - 10 EU]', ' [X = - 100 EU]', ' [X = - 1 000 EU]', ' [X = - 10 000 EU]', ' [X = - 100 000 EU]', 'Y=2 EU', 'Y=20 EU', 'Y=200 EU', 'Y=2 000 EU', 'Y=20 000 EU', 'Y=200 000 EU', 'Y= - 2 EU', 'Y= - 20 EU', 'Y= - 200 EU', 'Y= - 2 000 EU', 'Y= - 20 000 EU', 'Y= - 200 000 EU']].dropna()

# Calculate the average inter-item correlation for each division
average_inter_item_correlations = []
for column in work_around.columns:
    division_corr = work_around.drop(column, axis=1).corr().values
    average_corr = division_corr.mean()
    average_inter_item_correlations.append(average_corr)

# Print the average inter-item correlation for each division
for i, avg_corr in enumerate(average_inter_item_correlations):
    print(f"Division {i+1}: Average Inter-Item Correlation = {avg_corr:.2f}")


# In[106]:


import pandas as pd
import pingouin as pg

# Subset the DataFrame to include only the columns of interest and drop rows with missing values
subset_df = df[[' [X = 1 EU]', ' [X = 10 EU]', ' [X = 100 EU]', ' [X = 1 000 EU]', ' [X = 10 000 EU]', ' [X = 100 000 EU]', ' [X = - 1 EU ]', ' [X = - 10 EU]', ' [X = - 100 EU]', ' [X = - 1 000 EU]', ' [X = - 10 000 EU]', ' [X = - 100 000 EU]', 'Y=2 EU', 'Y=20 EU', 'Y=200 EU', 'Y=2 000 EU', 'Y=20 000 EU', 'Y=200 000 EU', 'Y= - 2 EU', 'Y= - 20 EU', 'Y= - 200 EU', 'Y= - 2 000 EU', 'Y= - 20 000 EU', 'Y= - 200 000 EU']].dropna()

# Calculate the Cronbach's alpha coefficient
alpha = pg.cronbach_alpha(subset_df)

alpha[0]


# In[107]:


set1 =  df[[' [X = 1 EU]', ' [X = 10 EU]', ' [X = 100 EU]', ' [X = 1 000 EU]', ' [X = 10 000 EU]', ' [X = 100 000 EU]']].dropna()
set2 = df[[' [X = - 1 EU ]', ' [X = - 10 EU]', ' [X = - 100 EU]', ' [X = - 1 000 EU]', ' [X = - 10 000 EU]', ' [X = - 100 000 EU]']].dropna()
set3 = df[['Y=2 EU', 'Y=20 EU', 'Y=200 EU', 'Y=2 000 EU', 'Y=20 000 EU', 'Y=200 000 EU']].dropna()
set4 = df[['Y= - 2 EU', 'Y= - 20 EU', 'Y= - 200 EU', 'Y= - 2 000 EU', 'Y= - 20 000 EU', 'Y= - 200 000 EU']].dropna()
set3_2 = q18
set4_2 = q24


# In[108]:


all_dfs = [set1,set2,set3, set4, set3_2, set4_2]


# # Risk Value

# In[109]:


pd.options.display.float_format = '{:,.2f}'.format
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import xgboost as xgb
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score


# ## Data cleaning

# In[110]:


data = pd.DataFrame()


# In[111]:


translation_dict = {
    'Икономика и бизнес  //  Economics and Business': 'Economics and Business',
    'Технически науки  //  Technical sciences': 'Technical sciences',
    'Хуманитарни науки // Humanities': 'Humanities',
    'Правни и политически науки': 'Law and Political Science',
    'Изкуства': 'Arts',
    'MBA': 'MBA',
    'Икономика': 'Economics',
    'Информатика и компютърни науки.': 'Informatics and Computer Science',
    'Социални науки  //  Social Sciences': 'Social Sciences',
    'Математика  //  Mathematics': 'Mathematics',
    'Администрация и управление': 'Administration and Management',
    'Природни науки': 'Natural Sciences',
    'Сигурност и отбрана': 'Security and Defense',
    'теология': 'Theology',
    'Здравеопазване и спорт': 'Healthcare and Sports',
    'Педагогически науки': 'Pedagogical Sciences',
    'непрофилирано': 'Unspecified',
    'дизайн': 'Design',
    'айляк': 'Laid-back attitude',
    'Маркетинг': 'Marketing',
    'Религия': 'Religion',
    'Аграрни науки': 'Agricultural Sciences',
    'Управление и администрация': 'Management and Administration',
    'Бизнес китайски ': 'Business Chinese',
    'Бизнес администрация': 'Business Administration',
    'Езикова': 'Language Studies',
    'Чужди езици': 'Foreign Languages',
    'Англ.ез': 'English Language',
    'чужди езици': 'Foreign Languages',
    'Природни науки; Икономика и бизнес; Социални науки;': 'Natural Sciences; Economics and Business; Social Sciences;',
    'Спорт': 'Sports',
    'бизнес администрация': 'Business Administration',
    'Хазарт': 'Gambling',
    'Кино техника ': 'Cinema Technology',
    'Туризъм': 'Tourism',
    'Езици': 'Languages',
    'Турезъм': 'Tourism',
    'транспорт': 'Transportation',
    'а по душа, спорта ме влече :)': 'By soul, I am attracted to sports :)',
    'Езици ': 'Languages',
    'ОРГАНИЗАЦИЯ НА ХОТИЛИЕРСТВОТО -ТУРИЗЪМ': 'Hospitality Organization - Tourism',
    'Авиация': 'Aviation',
    'Информатика и компютърни науки  //  Informatics and computer science': 'Informatics and Computer Science',
    'Завършила съм АЕГ, учила съм биология 2 години и сега уча Публична администрация първа година ': 'I have completed secondary education, studied biology for 2 years, and now I am studying Public Administration in the first year',
    'Хранително - вкусова промишленост': 'Food and Flavor Industry'
}

categories_dict = {
    'Economics and Business': 'Social Sciences',
    'Technical sciences': 'Science and Technology',
    'Humanities': 'Arts and Humanities',
    'Law and Political Science': 'Social Sciences',
    'Arts': 'Arts and Humanities',
    'MBA': 'Business and Management',
    'Economics': 'Social Sciences',
    'Informatics and Computer Science': 'Science and Technology',
    'Social Sciences': 'Social Sciences',
    'Mathematics': 'Science and Technology',
    'Administration and Management': 'Business and Management',
    'Natural Sciences': 'Science and Technology',
    'Security and Defense': 'Social Sciences',
    'Theology': 'Religion',
    'Healthcare and Sports': 'Health and Sports',
    'Pedagogical Sciences': 'Education',
    'Unspecified': 'Other',
    'Design': 'Arts and Humanities',
    'Laid-back attitude': 'Other',
    'Marketing': 'Business and Management',
    'Religion': 'Religion',
    'Agricultural Sciences': 'Science and Technology',
    'Management and Administration': 'Business and Management',
    'Business Chinese': 'Language Studies',
    'Business Administration': 'Business and Management',
    'Language Studies': 'Language Studies',
    'Foreign Languages': 'Language Studies',
    'English Language': 'Language Studies',
    'Natural Sciences; Economics and Business; Social Sciences;': 'Multidisciplinary',
    'Sports': 'Health and Sports',
    'Gambling': 'Other',
    'Cinema Technology': 'Arts and Humanities',
    'Tourism': 'Hospitality and Tourism',
    'Languages': 'Language Studies',
    'Transportation': 'Engineering and Transportation',
    'By soul, I am attracted to sports :)': 'Health and Sports',
    'Hospitality Organization - Tourism': 'Hospitality and Tourism',
    'Aviation': 'Engineering and Transportation',
    'I have completed secondary education, studied biology for 2 years, and now I am studying Public Administration in the first year': 'Other',
    'Food and Flavor Industry': 'Other',
    'I have completed secondary education, studied biology for 2 years, and now I am studying Public Administration in the first year':'Other',
    'Transportation, By soul, I am attracted to sports :)':'Engineering and Transportation',
    'Natural Sciences; Economics and Business; Social Sciences;':'Multidisciplinary'
    
}

city_dict = {
    'столица //  capital city': 'Capital city',
    'голям град  //  city': 'Big city',
    'село  //  village': 'Village',
    'чужбина': 'Abroad',
    'малък град  //  town': 'Small town',
    'London': 'Abroad',
    'Германия': 'Abroad',
    'Berlin': 'Abroad',
    'малък град, чужбина': 'Abroad',
    'Голям град в чужда държава': 'Abroad',
    'Пловидв': 'Big city',
    'предимно живея в столицата, но през последните няколко седмици живея в малък град': 'Capital city',
    'Тенерифе': 'Abroad'
}

jobs = {
    'служител  //  employee': 'Employee',
    'не работещ  //  unemployed': 'Unemployed',
    'експерт  //  expert': 'Expert',
    'мениджър  //  manager': 'Manager',
    'някои ръководни функции  //  some managerial functions': 'Managerial Role',
    'Студент': 'Student',
    'ски учител': 'Ski Instructor',
    'самонает': 'Self-employed',
    'самоосигуряващ се (свободна професия)': 'Self-employed',
    'студент': 'Student',
    'CTO': 'CTO',
    'пенсионер': 'Retired',
    'Freelancer': 'Freelancer',
    'Юрист': 'Lawyer',
    'Самонает': 'Self-employed',
    'фриленсър - експерт': 'Freelancer',
    'собственик': 'Business Owner',
    'работещ на свободна практика': 'Self-employed',
    'балъче': 'Dancer',
    'Студентка': 'Student',
    'ученик': 'Student',
    'свободно практикуващ': 'Self-employed',
    'стажант': 'Intern',
    'пастор': 'Pastor',
    'майка': 'Stay-at-home Parent',
    'Счетоводител ': 'Accountant',
    'Специалист': 'Specialist',
    'assistant manager': 'Assistant Manager',
    'Докторант': 'PhD Student',
    'student': 'Student',
    'Съзтезател': 'Competitor',
    'свободна професия': 'Self-employed',
    'Студент в майчинство ': 'Student Parent',
    'Стажант': 'Intern',
    'студент, зает на непълен работен ден': 'Part-time Student',
    'Учащ': 'Student',
    'Главен специалист в държавна администрация': 'Government Specialist',
    'Медицински секретар': 'Medical Secretary',
    'Студент ': 'Student',
    'майчинство': 'Parent',
    'стаж': 'Internship',
    'Работещ на свободна практика': 'Self-employed',
    'echange student': 'Exchange Student',
    'Предприемач//entrepreneur': 'Entrepreneur',
    'Student ': 'Student',
    'Собствен бизнес': 'Business Owner'
}

jobs2 = {
    'Employee': 'Employee',
    'Unemployed': 'Unemployed',
    'Expert': 'Expert',
    'Manager': 'Manager',
    'Managerial Role': 'Manager',
    'Student': 'Student',
    'Intern': 'Intern',
    'Self-employed': 'Self-employed',
    'Freelancer': 'Self-employed',
    'Business Owner': 'Self-employed',
    'Government Specialist': 'Specialist',
    'Student Parent': 'Other',
    'Part-time Student': 'Student',
    'Parent': 'Other',
    'Medical Secretary': 'Specialist',
    'PhD Student': 'Student',
    'Internship': 'Intern',
    'Exchange Student': 'Student',
    'Competitor': 'Student',
    'Pastor': 'Other',
    'Assistant Manager': 'Manager',
    'Specialist': 'Specialist',
    'Accountant': 'Specialist',
    'Stay-at-home Parent': 'Other',
    'Dancer': 'Specialist',
    'Lawyer': 'Other',
    'Retired': 'Other',
    'CTO': 'Specialist',
    'Ski Instructor': 'Other',
    'Entrepreneur': 'Self-employed'
}

activity = {
    'Услуги  //  Services': 'Services',
    'Информационни технологии': 'Information Technology',
    'Финансова дейност // Financial activity': 'Financial Activity',
    'Производство // Production': 'Production',
    'Търговия  //  Trade': 'Trade',
    'Образование // education': 'Education',
    'Туризъм': 'Tourism',
    'Маркетинг и продажби': 'Marketing and Sales',
    'Изследователска и развойна дейност': 'Research and Development',
    'Транспорт // transport': 'Services',
    'Консултантски услуги': 'Services',
    'Човешки Ресурси': 'Services',
    'project management': 'Services',
    'Маркетинг и реклама': 'Marketing and Sales',
    'изкарвам толкова понеже знам немски и английски, иначе... ': 'Other',
    'администрация': 'Services',
    'държавна администрация': 'Services',
    'правни науки': 'Services',
    'пенсионер': 'Other',
    'религия': 'Other',
    'Човешки ресурси': 'Services',
    'HR': 'Services',
    'Знание-интензивен аутсорсинг': 'Services',
    'студент': 'Education',
    'Юридическа дейност': 'Services',
    'изкуство и туризъм': 'Tourism',
    'реклама': 'Marketing and Sales',
    'Клинични проучвания': 'Research and Development',
    'Здвавни грижи': 'Services',
    'Маркетинг': 'Marketing and Sales',
    'Склад': 'Services',
    '...': 'Other',
    'Строителство': 'Production',
    'Инженеринг': 'Production',
    'Стройтелство': 'Production',
    'просветителна': 'Services',
    'Телекомуникации': 'Services',
    'Администрация': 'Services',
    'Застраховане': 'Financial Activity',
    'Бизнес': 'Services',
    'Недвижими имоти': 'Services',
    'Не работя': 'Unemployed',
    'Няма такава': 'Other',
    'продажби ': 'Marketing and Sales',
    'Покер играч': 'Other',
    'медии': 'Services',
    'PR': 'Services',
    'Хотелиерство и ресторантьорство': 'Tourism',
    'sport': 'Sports',
    'инвестиции': 'Financial Activity',
    'Спорт': 'Sports',
    'застраховател': 'Financial Activity',
    'Охранителен сектор': 'Services',
    'Фотография ': 'Services',
    'Бразработна съм': 'Services',
    'Физическа активност, спорт': 'Sports',
    'студент икономика и финанси ': 'Education',
    'студент по стопанско управление ': 'Education',
    'Военно дело': 'Services',
    'Публичен сектор (Сигурност)': 'Services',
    'пожарна безопасност': 'Services',
    'право': 'Services',
    'Електроразпределение не енергия': 'Services',
    'Енергетика': 'Energy',
    'Електроразпределение': 'Services',
    'Разпределение на ел. енергия': 'Services',
    'Рециклиране': 'Services',
    'Social Media Agent': 'Services',
    'учащ': 'Education',
    'материално право и търговия': 'Services',
    'Анализ на данни': 'Services',
    'сигурност и отбрана': 'Security and Defense',
    'Административна/политика': 'Services',
    ' Системата за събиране на местни данъци и такси': 'Services',
    'Сигурност': 'Security and Defense',
    'В момента съм студентка и не работя никъде.': 'Education',
    'Студент съм, не работя никъде ': 'Education',
    'БАНКОВИЯТ СЕКТОР': 'Financial Activity',
    'Нямам': 'Other',
    'не работя в момента': 'Unemployed',
    'служител в държавна администрация': 'Services',
    'хотелиерство': 'Tourism',
    'Медицински служител': 'Services',
    'Логистика': 'Services',
    'Онлайн плащания': 'Services',
    'ОБЩИНСКА АДМИНИСТРАЦИЯ': 'Services',
    'unemployed': 'Unemployed',
    'Информационни технологии //  Information Technology': 'Information Technology',
    'Нищо': 'Other',
    'не работя': 'Unemployed',
    'Финанси, услуги и търговия': 'Financial Activity',
    'Real estate agent': 'Services',
    'Държавна администрация': 'Services'
}


# In[112]:


# Mapping values to a new column
data['Gender'] = (df_raw['Какъв е Вашият пол?'].map({'Жена / Female': 0, 'Мъж / Male': 1}))
data['Age'] = df_raw['Каква е Вашата възраст в години?'].copy()
data['YearsOfEducation'] = df_raw['Колко общо години образование имате от първи клас включително? // How many years of first grade education do you have in total?'].copy()
data['MainAreaOfEducation'] = (df_raw['В какво професионално направление е Вашето най-съществено образование?'].map(translation_dict)).map(categories_dict)
data['Income'] = df_raw['Какви са Вашите приблизителни средно месечни доходи?'].copy()
data['AreaOfLiving'] = df_raw['В какво населено място живеете?'].map(city_dict)
data['Occupation'] = (df_raw['Каква е Вашата настояща позиция?'].map(jobs)).map(jobs2)
data['AreaOfExpertise'] = (df_raw['Каква е Вашата сфера на дейност?'].map(activity))
data[[' [X = 1 EU]', ' [X = 10 EU]',
       ' [X = 100 EU]', ' [X = 1 000 EU]', ' [X = 10 000 EU]',
       ' [X = 100 000 EU]', ' [X = - 1 EU ]', ' [X = - 10 EU]',
       ' [X = - 100 EU]', ' [X = - 1 000 EU]', ' [X = - 10 000 EU]',
       ' [X = - 100 000 EU]', 'Y=2 EU', 'Y=20 EU', 'Y=200 EU', 'Y=2 000 EU',
       'Y=20 000 EU', 'Y=200 000 EU', 'Y= - 2 EU', 'Y= - 20 EU', 'Y= - 200 EU',
       'Y= - 2 000 EU', 'Y= - 20 000 EU', 'Y= - 200 000 EU']] = df_raw[[' [X = 1 EU]', ' [X = 10 EU]',
       ' [X = 100 EU]', ' [X = 1 000 EU]', ' [X = 10 000 EU]',
       ' [X = 100 000 EU]', ' [X = - 1 EU ]', ' [X = - 10 EU]',
       ' [X = - 100 EU]', ' [X = - 1 000 EU]', ' [X = - 10 000 EU]',
       ' [X = - 100 000 EU]', 'Y=2 EU', 'Y=20 EU', 'Y=200 EU', 'Y=2 000 EU',
       'Y=20 000 EU', 'Y=200 000 EU', 'Y= - 2 EU', 'Y= - 20 EU', 'Y= - 200 EU',
       'Y= - 2 000 EU', 'Y= - 20 000 EU', 'Y= - 200 000 EU']]
data['Risk_AVG'] = df_raw['risk-result']


# ## Demos

# In[113]:


data['Gender'] = (df_raw['Какъв е Вашият пол?'].map({'Жена / Female': 0, 'Мъж / Male': 1}))
data['Age'] = df_raw['Каква е Вашата възраст в години?'].copy()
data['YearsOfEducation'] = df_raw['Колко общо години образование имате от първи клас включително? // How many years of first grade education do you have in total?'].copy()
data['MainAreaOfEducation'] = (df_raw['В какво професионално направление е Вашето най-съществено образование?'].map(translation_dict)).map(categories_dict)
data['AreaOfLiving'] = df_raw['В какво населено място живеете?'].map(city_dict)
data['Occupation'] = (df_raw['Каква е Вашата настояща позиция?'].map(jobs)).map(jobs2)
data['AreaOfExpertise'] = (df_raw['Каква е Вашата сфера на дейност?'].map(activity))


# In[114]:


demo_df = data[['Gender','Age','YearsOfEducation',
                   'MainAreaOfEducation','AreaOfLiving','Occupation',
                   'AreaOfExpertise',' [X = 1 EU]', ' [X = 10 EU]',
       ' [X = 100 EU]', ' [X = 1 000 EU]', ' [X = 10 000 EU]',
       ' [X = 100 000 EU]', ' [X = - 1 EU ]', ' [X = - 10 EU]',
       ' [X = - 100 EU]', ' [X = - 1 000 EU]', ' [X = - 10 000 EU]',
       ' [X = - 100 000 EU]', 'Y=2 EU', 'Y=20 EU', 'Y=200 EU', 'Y=2 000 EU',
       'Y=20 000 EU', 'Y=200 000 EU', 'Y= - 2 EU', 'Y= - 20 EU', 'Y= - 200 EU',
       'Y= - 2 000 EU', 'Y= - 20 000 EU', 'Y= - 200 000 EU','Risk_AVG']] 


# In[115]:


demo_df.dropna(how='all', subset=demo_df.columns.difference(['Risk_AVG']), inplace=True)


# In[116]:


demo_df


# In[117]:


demo_df.columns


# ## Stepwise demographics vs all

# In[ ]:





# In[118]:


import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder

# Define the columns of interest
columns_categorical = ['Gender','MainAreaOfEducation','AreaOfLiving', 'Occupation', 'AreaOfExpertise',
                      ' [X = 1 EU]',
       ' [X = 10 EU]', ' [X = 100 EU]', ' [X = 1 000 EU]', ' [X = 10 000 EU]',
       ' [X = 100 000 EU]', ' [X = - 1 EU ]', ' [X = - 10 EU]',
       ' [X = - 100 EU]', ' [X = - 1 000 EU]', ' [X = - 10 000 EU]',
       ' [X = - 100 000 EU]']

columns_numerical = ['Age', 'YearsOfEducation','Y=2 EU', 'Y=20 EU', 'Y=200 EU', 'Y=2 000 EU',
       'Y=20 000 EU', 'Y=200 000 EU', 'Y= - 2 EU', 'Y= - 20 EU', 'Y= - 200 EU',
       'Y= - 2 000 EU', 'Y= - 20 000 EU', 'Y= - 200 000 EU']


target = demo_df[['Risk_AVG']]

classifiers  = [' [X = 1 EU]',
       ' [X = 10 EU]', ' [X = 100 EU]', ' [X = 1 000 EU]', ' [X = 10 000 EU]',
       ' [X = 100 000 EU]', ' [X = - 1 EU ]', ' [X = - 10 EU]',
       ' [X = - 100 EU]', ' [X = - 1 000 EU]', ' [X = - 10 000 EU]',
       ' [X = - 100 000 EU]']


# Define the feature columns dictionary
feature_columns = {'categorical': columns_categorical, 'numerical': columns_numerical}


# In[119]:


# Assuming demo_df is your DataFrame and columns_categorical and columns_numerical are the respective lists of column names
categorical_df = demo_df[columns_categorical].fillna(demo_df[columns_categorical].mode().iloc[0])
numerical_df = demo_df[columns_numerical].fillna(demo_df[columns_numerical].mean())

# Perform one-hot encoding on the categorical columns
one_hot_encoded_df = pd.get_dummies(categorical_df)

# Concatenate the one-hot encoded DataFrame with the numerical columns
new_df = pd.concat([one_hot_encoded_df, numerical_df], axis=1)


# In[120]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression

def stepwise_regression(X, y, threshold_in=0.05, threshold_out=0.10):
    included = []
    excluded = list(X.columns)
    
    while True:
        changed = False
        # Forward step
        new_pval = pd.Series(index=excluded)
        for feature in excluded:
            model = LinearRegression()
            X_included = X[included + [feature]]
            model.fit(X_included, y)
            y_pred = model.predict(X_included)
            pvals = f_regression(X_included, y)[1]
            new_pval[feature] = pvals[-1]
        
        best_feature = new_pval.idxmin()
        if new_pval[best_feature] < threshold_in:
            included.append(best_feature)
            excluded.remove(best_feature)
            changed = True
        
        # Backward step
        model = LinearRegression()
        X_included = X[included]
        model.fit(X_included, y)
        y_pred = model.predict(X_included)
        pvals = f_regression(X_included, y)[1]
        worst_feature = pvals.argmax()
        if pvals[worst_feature] > threshold_out:
            excluded.append(included[worst_feature])
            included.remove(included[worst_feature])
            changed = True
        
        if not changed:
            break
    
    return included

# Example usage
# X is the input features, y is the target variable
# You should replace X and y with your own data

# Assuming X is a pandas DataFrame with the input features and y is a pandas Series with the target variable
selected_features = stepwise_regression(new_df, target.values.ravel())
print("Selected features:", selected_features)


# In[121]:


new_df[['Y= - 2 000 EU', 'Y= - 200 EU', 'Y= - 20 EU', 'Y= - 2 EU', 'Y= - 20 000 EU', 'Y= - 200 000 EU', 'Y=200 EU', 'Y=2 000 EU', 'Y=20 000 EU', 'Y=20 EU', 'Y=200 000 EU', 'Y=2 EU', ' [X = - 100 000 EU]', 'AreaOfExpertise_Research and Development', ' [X = - 10 000 EU]', ' [X = 100 EU]', 'AreaOfLiving_Big city', ' [X = 10 EU]', 'Occupation_Manager', ' [X = - 1 000 EU]', 'Occupation_Expert', ' [X = 1 EU]', ' [X = - 100 EU]']]


# In[122]:


import re

# Step 1: Define the features and target variable
features = new_df  # Exclude the target variable from features
features.columns = [re.sub(r'[\[\]<]', '', feature) for feature in features.columns]
# target = target = 
target = target


# In[123]:


# Step 4: Create and fit the XGBoost regressor
regressor = XGBRegressor()
regressor.fit(features, target)


# In[124]:


# Step 6: Predict missing values using the fitted regressor
y_pred = (regressor.predict(features))


# In[125]:


# Step 1: Retrieve the feature importances from the trained regressor
importance = regressor.feature_importances_

# Step 2: Create a DataFrame to store the feature importances
feature_importances = pd.DataFrame({'Feature': features.columns, 'Importance': importance})

# Step 3: Sort the feature importances in descending order
feature_importances = feature_importances.sort_values('Importance', ascending=False).reset_index(drop=True)

# Step 4: Display the feature importances in a DataFrame
print(feature_importances)

# Step 5: Plot the feature importances in a bar chart
feature_importances.plot(kind='barh', x='Feature', y='Importance', figsize=(10, 6))
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.show()


# In[126]:


feature_importances.sort_values('Importance', ascending=False)[:20]


# ## Stepwise Demographics

# In[127]:


f_s = new_df[['Gender',
       'MainAreaOfEducation_Arts and Humanities',
       'MainAreaOfEducation_Business and Management',
       'MainAreaOfEducation_Education',
       'MainAreaOfEducation_Engineering and Transportation',
       'MainAreaOfEducation_Health and Sports',
       'MainAreaOfEducation_Hospitality and Tourism',
       'MainAreaOfEducation_Language Studies',
       'MainAreaOfEducation_Multidisciplinary', 
        'MainAreaOfEducation_Other',
       'MainAreaOfEducation_Religion',
       'MainAreaOfEducation_Science and Technology',
       'MainAreaOfEducation_Social Sciences', 
       'AreaOfLiving_Abroad',
       'AreaOfLiving_Big city', 
       'AreaOfLiving_Capital city',
       'AreaOfLiving_Small town', 
       'AreaOfLiving_Village',
       'Occupation_Employee', 
        'Occupation_Expert', 
        'Occupation_Intern',
       'Occupation_Manager', 
        'Occupation_Other', 
        'Occupation_Self-employed',
       'Occupation_Specialist', 
        'Occupation_Student', 
        'Occupation_Unemployed',
       'AreaOfExpertise_Education', 
        'AreaOfExpertise_Energy',
       'AreaOfExpertise_Financial Activity',
       'AreaOfExpertise_Information Technology',
       'AreaOfExpertise_Marketing and Sales', 
        'AreaOfExpertise_Other',
       'AreaOfExpertise_Production',
       'AreaOfExpertise_Research and Development',
       'AreaOfExpertise_Security and Defense', 
        'AreaOfExpertise_Services',
       'AreaOfExpertise_Sports', 
        'AreaOfExpertise_Tourism',
       'AreaOfExpertise_Trade', 
        'AreaOfExpertise_Unemployed', 
        'Age',
       'YearsOfEducation']]

tts = [' X = 1 EU', ' X = 10 EU', ' X = 100 EU', ' X = 1 000 EU',
       ' X = 10 000 EU', ' X = 100 000 EU', ' X = - 1 EU ', ' X = - 10 EU',
       ' X = - 100 EU', ' X = - 1 000 EU', ' X = - 10 000 EU',
       ' X = - 100 000 EU','Y=2 EU', 'Y=20 EU', 'Y=200 EU', 'Y=2 000 EU',
       'Y=20 000 EU', 'Y=200 000 EU', 'Y= - 2 EU', 'Y= - 20 EU', 'Y= - 200 EU',
       'Y= - 2 000 EU', 'Y= - 20 000 EU', 'Y= - 200 000 EU']

new_df.columns


# In[128]:


def xgb(features, target):
    # Clean feature names
    features.columns = [re.sub(r'[\[\]<]', '', feature) for feature in features.columns]

    # Create and fit the XGBoost regressor
    regressor = XGBRegressor()
    regressor.fit(features, target)

    # Make predictions on the features
    predictions = regressor.predict(features)

    # Calculate the R2 score
    r2 = r2_score(target, predictions)
    features = pd.DataFrame({'Feature': features.columns, f'Importance': regressor.feature_importances_})

    return r2, features


# In[129]:


importances = [xgb(f_s,new_df[[target]])[1][['Importance']] for target in tts]
importance_df = pd.concat(importances, axis = 1)
importance_df.index = f_s.columns
importance_df.columns = tts


# In[130]:


importance_df #demographics for every model


# ## Feature Importance

# In[131]:


tts = [' X = 1 EU', ' X = 10 EU', ' X = 100 EU', ' X = 1 000 EU',
       ' X = 10 000 EU', ' X = 100 000 EU', ' X = - 1 EU ', ' X = - 10 EU',
       ' X = - 100 EU', ' X = - 1 000 EU', ' X = - 10 000 EU',
       ' X = - 100 000 EU','Y=2 EU', 'Y=20 EU', 'Y=200 EU', 'Y=2 000 EU',
       'Y=20 000 EU', 'Y=200 000 EU', 'Y= - 2 EU', 'Y= - 20 EU', 'Y= - 200 EU',
       'Y= - 2 000 EU', 'Y= - 20 000 EU', 'Y= - 200 000 EU']

target_df = new_df[tts]


# In[132]:


def split_df(target_df, column):
    return target_df.loc[:, :column]
 


# In[133]:


dffs = [split_df(target_df, columns) for columns in tts[1:]]


# In[134]:


def xgb2(df):
    # Clean feature names
    features = df.iloc[:,:-1]
    target = df.iloc[:,-1:]
    
    regressor = XGBRegressor()
    regressor.fit(features, target)
    predictions = regressor.predict(features)
    
    r2 = r2_score(predictions, target)
    
    importance = pd.DataFrame({'Feature': features.columns, f'Importance': regressor.feature_importances_})
    return r2,importance


# In[135]:


r2s = [xgb2(df)[0] for df in dffs]
features = [xgb2(df)[1] for df in dffs]


# In[136]:


r2_per_model = pd.DataFrame({'Target': tts[1:], 'R2': r2s})
r2_per_model


# In[137]:


f_df = pd.concat([f[['Importance']] for f in features],axis=1)
f_df.columns = tts[1:]
f_df.index = tts[:-1]
f_df


# In[138]:


data.describe()


# In[139]:


import plotly.graph_objects as go
import numpy as np

# Sample risk scores
risk_scores = data['Risk_AVG'].values

# Create a histogram with more bins
fig = go.Figure(data=go.Histogram(x=risk_scores, nbinsx=20))  # Increase nbinsx value for more bins

# Set chart title and axis labels
fig.update_layout(
    title={
        'text': "Risk Scores Histogram",
        'x': 0.5,  # Set the x position to center the title
        'xanchor': 'center',  # Anchor the title to the center
    },
    xaxis_title="Risk-result",
    yaxis_title="Count",
    plot_bgcolor='rgba(0,0,0,0)'  # Set background color to transparent
)

# Set the background color of the plot area to transparent
fig.update_xaxes(showgrid=False, zeroline=False, tickvals=np.arange(0, 1.1, 0.1), ticktext=[round(x, 2) for x in np.arange(0, 1.1, 0.1)])  # x-axis labels with 0.1 interval, rounded to 2 decimal places
fig.update_yaxes(showgrid=False, zeroline=False)  # Example for y-axis labels

# Show the plot
fig.show()


# In[140]:


data['Risk_AVG']


# In[141]:


# Compute correlations
correlations = data.corr()['Risk_AVG']


# In[142]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'data' is your DataFrame

# Compute Spearman correlations
correlations = data.corr(method='spearman')

# Remove NaN values (if any)
correlations = correlations.dropna()

# Create correlation plot
plt.figure(figsize=(10, 8))
sns.heatmap(correlations, annot=False, cmap='RdYlGn', center=0)

# Set plot title
plt.title('Spearman Correlation Plot')

# Remove x-axis and y-axis labels
plt.xlabel('')
plt.ylabel('')

# Show the plot
plt.show()


# In[143]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Assuming 'data' is your DataFrame

# Compute correlations
correlations = data.corr()

# Remove NaN values (if any)
correlations = correlations.dropna()

# Create a mask for upper triangular portion
mask = np.triu(np.ones_like(correlations, dtype=bool))

# Create correlation plot
plt.figure(figsize=(10, 8))
sns.heatmap(correlations, annot=False, cmap='RdYlGn', center=0, mask=mask)

# Set plot title
plt.title('Spearman Correlation Plot (Upper Triangular)')

# Remove x-axis and y-axis labels
plt.xlabel('')
plt.ylabel('')

# Show the plot
plt.show()


# # Data

# In[144]:


data.columns


# In[145]:


pg.cronbach_alpha(data[[' [X = 10 EU]', ' [X = 100 EU]', ' [X = 1 000 EU]', ' [X = 10 000 EU]',
       ' [X = 100 000 EU]', ' [X = - 1 EU ]', ' [X = - 10 EU]',
       ' [X = - 100 EU]', ' [X = - 1 000 EU]', ' [X = - 10 000 EU]',
       ' [X = - 100 000 EU]']].dropna())


# In[146]:


pg.cronbach_alpha(data[['Y=20 000 EU', 'Y=200 000 EU', 'Y= - 2 EU', 'Y= - 20 EU', 'Y= - 200 EU',
       'Y= - 2 000 EU', 'Y= - 20 000 EU', 'Y= - 200 000 EU']].dropna())


# In[147]:


pg.cronbach_alpha(data[[' [X = 1 EU]',
       ' [X = 10 EU]', ' [X = 100 EU]', ' [X = 1 000 EU]', ' [X = 10 000 EU]',
       ' [X = 100 000 EU]', ' [X = - 1 EU ]', ' [X = - 10 EU]',
       ' [X = - 100 EU]', ' [X = - 1 000 EU]', ' [X = - 10 000 EU]',
       ' [X = - 100 000 EU]', 'Y=2 EU', 'Y=20 EU', 'Y=200 EU', 'Y=2 000 EU',
       'Y=20 000 EU', 'Y=200 000 EU', 'Y= - 2 EU', 'Y= - 20 EU', 'Y= - 200 EU',
       'Y= - 2 000 EU', 'Y= - 20 000 EU', 'Y= - 200 000 EU']].dropna())


# In[148]:


data[[' [X = 1 EU]',
       ' [X = 10 EU]', ' [X = 100 EU]', ' [X = 1 000 EU]', ' [X = 10 000 EU]',
       ' [X = 100 000 EU]', ' [X = - 1 EU ]', ' [X = - 10 EU]',
       ' [X = - 100 EU]', ' [X = - 1 000 EU]', ' [X = - 10 000 EU]',
       ' [X = - 100 000 EU]', 'Y=2 EU', 'Y=20 EU', 'Y=200 EU', 'Y=2 000 EU',
       'Y=20 000 EU', 'Y=200 000 EU', 'Y= - 2 EU', 'Y= - 20 EU', 'Y= - 200 EU',
       'Y= - 2 000 EU', 'Y= - 20 000 EU', 'Y= - 200 000 EU']].dropna()


# In[149]:


import pingouin as pg

# Select the desired columns and drop rows with missing values
data_selected = data[['Y= - 2 EU', 'Y= - 20 EU', 'Y= - 200 EU', 'Y= - 2 000 EU', 'Y= - 20 000 EU', 'Y= - 200 000 EU']].dropna()
# data_selected = data[['Y=2 EU', 'Y=20 EU', 'Y=200 EU', 'Y=2 000 EU',
#        'Y=20 000 EU', 'Y=200 000 EU']]
# Calculate Cronbach's alpha
alpha = pg.cronbach_alpha(data_selected)

print("Cronbach's alpha:", alpha)


# In[150]:


import pandas as pd
import pingouin as pg

# Select the desired columns
columns = ['Y= - 2 EU', 'Y= - 20 EU', 'Y= - 200 EU', 'Y= - 2 000 EU', 'Y= - 20 000 EU', 'Y= - 200 000 EU']
# columns = ['Y=2 EU', 'Y=20 EU', 'Y=200 EU', 'Y=2 000 EU',
#        'Y=20 000 EU', 'Y=200 000 EU']

# Create an empty DataFrame to store the results
results = pd.DataFrame(columns=['Removed Column', 'Remaining Columns', "Cronbach's Alpha"])

# Iterate over each column and calculate Cronbach's alpha for the remaining variables
for column in columns:
    remaining_columns = [col for col in columns if col != column]
    data_selected = data[remaining_columns].dropna()
    alpha = pg.cronbach_alpha(data_selected)
    results = results.append({'Removed Column': column,
                              'Remaining Columns': ', '.join(remaining_columns),
                              "Cronbach's Alpha": alpha},
                             ignore_index=True)

# Print the results
results


# In[151]:


data.columns


# In[152]:


def transform_column(df, column_name, n):
    df[column_name + '_transformed'] = df[column_name].apply(lambda x: 1 if x > n else 0 if x == n else -1)
    return df


# In[153]:


data = transform_column(data, 'Y=2 EU', n = 1)
data = transform_column(data, 'Y=20 EU', n = 10)
data = transform_column(data, 'Y=200 EU', n = 100)
data = transform_column(data, 'Y=2 000 EU', n = 1000)
data = transform_column(data, 'Y=20 000 EU', n = 10000)
data = transform_column(data, 'Y=200 000 EU', n = 100000)

data = transform_column(data, 'Y= - 2 EU', n = -1)
data = transform_column(data, 'Y= - 20 EU', n = -10)
data = transform_column(data, 'Y= - 200 EU', n = -100)
data = transform_column(data, 'Y= - 2 000 EU', n = -1000)
data = transform_column(data, 'Y= - 20 000 EU', n = -10000)
data = transform_column(data, 'Y= - 200 000 EU', n = -100000)


# In[154]:


data.columns


# In[155]:


pg.cronbach_alpha(data[['Y=2 EU', 'Y=20 EU', 'Y=200 EU', 'Y=2 000 EU',
      'Y=20 000 EU', 'Y=200 000 EU']])


# In[156]:


pg.cronbach_alpha(data[['Y=2 EU_transformed', 'Y=20 EU_transformed', 'Y=200 EU_transformed',
      'Y=2 000 EU_transformed', 'Y=20 000 EU_transformed',
      'Y=200 000 EU_transformed']])


# In[157]:


import pandas as pd
import pingouin as pg

# Select the desired columns
columns = ['Y=2 EU_transformed', 'Y=20 EU_transformed', 'Y=200 EU_transformed',
       'Y=2 000 EU_transformed', 'Y=20 000 EU_transformed',
       'Y=200 000 EU_transformed']
# columns = ['Y=2 EU', 'Y=20 EU', 'Y=200 EU', 'Y=2 000 EU',
#        'Y=20 000 EU', 'Y=200 000 EU']

# Create an empty DataFrame to store the results
results = pd.DataFrame(columns=['Removed Column', 'Remaining Columns', "Cronbach's Alpha"])

# Iterate over each column and calculate Cronbach's alpha for the remaining variables
for column in columns:
    remaining_columns = [col for col in columns if col != column]
    data_selected = data[remaining_columns].dropna()
    alpha = pg.cronbach_alpha(data_selected)
    results = results.append({'Removed Column': column,
                              'Remaining Columns': ', '.join(remaining_columns),
                              "Cronbach's Alpha": alpha},
                             ignore_index=True)

# Print the results
results


# In[158]:


pg.cronbach_alpha(data[['Y= - 2 EU', 'Y= - 20 EU', 'Y= - 200 EU',
      'Y= - 2 000 EU', 'Y= - 20 000 EU', 'Y= - 200 000 EU']])


# In[159]:


pg.cronbach_alpha(data[['Y= - 2 EU_transformed',
      'Y= - 20 EU_transformed', 'Y= - 200 EU_transformed',
      'Y= - 2 000 EU_transformed', 'Y= - 20 000 EU_transformed',
      'Y= - 200 000 EU_transformed']])


# In[160]:


import pandas as pd
import pingouin as pg

# Select the desired columns
columns = ['Y= - 2 EU_transformed',
       'Y= - 20 EU_transformed', 'Y= - 200 EU_transformed',
       'Y= - 2 000 EU_transformed', 'Y= - 20 000 EU_transformed',
       'Y= - 200 000 EU_transformed']
# columns = ['Y=2 EU', 'Y=20 EU', 'Y=200 EU', 'Y=2 000 EU',
#        'Y=20 000 EU', 'Y=200 000 EU']

# Create an empty DataFrame to store the results
results = pd.DataFrame(columns=['Removed Column', 'Remaining Columns', "Cronbach's Alpha"])

# Iterate over each column and calculate Cronbach's alpha for the remaining variables
for column in columns:
    remaining_columns = [col for col in columns if col != column]
    data_selected = data[remaining_columns].dropna()
    alpha = pg.cronbach_alpha(data_selected)
    results = results.append({'Removed Column': column,
                              'Remaining Columns': ', '.join(remaining_columns),
                              "Cronbach's Alpha": alpha},
                             ignore_index=True)

# Print the results
results


# In[161]:


data.columns


# In[162]:


pg.cronbach_alpha(data[[' [X = 1 EU]',
       ' [X = 10 EU]', ' [X = 100 EU]', ' [X = 1 000 EU]', ' [X = 10 000 EU]',
       ' [X = 100 000 EU]', ' [X = - 1 EU ]', ' [X = - 10 EU]',
       ' [X = - 100 EU]', ' [X = - 1 000 EU]', ' [X = - 10 000 EU]',
       ' [X = - 100 000 EU]', 
                       'Y=2 EU_transformed', 'Y=20 EU_transformed', 'Y=200 EU_transformed',
       'Y=2 000 EU_transformed', 'Y=20 000 EU_transformed',
       'Y=200 000 EU_transformed', 'Y= - 2 EU_transformed',
       'Y= - 20 EU_transformed', 'Y= - 200 EU_transformed',
       'Y= - 2 000 EU_transformed', 'Y= - 20 000 EU_transformed',
       'Y= - 200 000 EU_transformed']])


# In[163]:


final = data[[' [X = 1 EU]',
       ' [X = 10 EU]', ' [X = 100 EU]', ' [X = 1 000 EU]', ' [X = 10 000 EU]',
       ' [X = 100 000 EU]', ' [X = - 1 EU ]', ' [X = - 10 EU]',
       ' [X = - 100 EU]', ' [X = - 1 000 EU]', ' [X = - 10 000 EU]',
       ' [X = - 100 000 EU]', 
                       'Y=2 EU_transformed', 'Y=20 EU_transformed', 'Y=200 EU_transformed',
       'Y=2 000 EU_transformed', 'Y=20 000 EU_transformed',
       'Y=200 000 EU_transformed', 'Y= - 2 EU_transformed',
       'Y= - 20 EU_transformed', 'Y= - 200 EU_transformed',
       'Y= - 2 000 EU_transformed', 'Y= - 20 000 EU_transformed',
       'Y= - 200 000 EU_transformed']]


# In[ ]:





# In[164]:


# Compute pairwise correlation
corr = final.corr().abs()

# Set a high correlation threshold
corr_threshold = 0.95

# Keep only columns with at least one correlation above the threshold
relevant_columns = corr.columns[(corr > corr_threshold).any()].tolist()

# From each group of correlated columns, choose one representative
best_subset = []
for col in relevant_columns:
    if all(corr[col][best_subset] < corr_threshold):
        best_subset.append(col)

# Compute Cronbach's alpha for the best subset
best_alpha = pg.cronbach_alpha(final[best_subset])

print("Best combination of columns:", best_subset)
print("Cronbach's alpha:", best_alpha)


# In[165]:


data[[' [X = 1 EU]',
       ' [X = 10 EU]', ' [X = 100 EU]', ' [X = 1 000 EU]', ' [X = 10 000 EU]',
       ' [X = 100 000 EU]', ' [X = - 1 EU ]', ' [X = - 10 EU]',
       ' [X = - 100 EU]', ' [X = - 1 000 EU]', ' [X = - 10 000 EU]',
       ' [X = - 100 000 EU]', 
                       'Y=2 EU_transformed', 'Y=20 EU_transformed', 'Y=200 EU_transformed',
       'Y=2 000 EU_transformed', 'Y=20 000 EU_transformed',
       'Y=200 000 EU_transformed', 'Y= - 2 EU_transformed',
       'Y= - 20 EU_transformed', 'Y= - 200 EU_transformed',
       'Y= - 2 000 EU_transformed', 'Y= - 20 000 EU_transformed',
       'Y= - 200 000 EU_transformed']]


# In[166]:


pg.cronbach_alpha(data[[' [X = 1 EU]',
       ' [X = 10 EU]', ' [X = 100 EU]', ' [X = 1 000 EU]', ' [X = 10 000 EU]',
       ' [X = 100 000 EU]']])


# In[167]:


pg.cronbach_alpha(data[[' [X = 1 EU]',
       ' [X = 10 EU]', ' [X = 100 EU]', ' [X = 1 000 EU]', ' [X = 10 000 EU]',
       ' [X = 100 000 EU]', ' [X = - 1 EU ]', ' [X = - 10 EU]',
       ' [X = - 100 EU]', ' [X = - 1 000 EU]', ' [X = - 10 000 EU]',
       ' [X = - 100 000 EU]',]])


# In[168]:


pg.cronbach_alpha(data[[' [X = 1 EU]',
       ' [X = 10 EU]', ' [X = 100 EU]', ' [X = 1 000 EU]', ' [X = 10 000 EU]',
       ' [X = 100 000 EU]', ' [X = - 1 EU ]', ' [X = - 10 EU]',
       ' [X = - 100 EU]', ' [X = - 1 000 EU]', ' [X = - 10 000 EU]',
       ' [X = - 100 000 EU]', 
                       'Y=2 EU_transformed', 'Y=20 EU_transformed', 'Y=200 EU_transformed',
       'Y=2 000 EU_transformed', 'Y=20 000 EU_transformed',
       'Y=200 000 EU_transformed', 'Y= - 2 EU_transformed',
       'Y= - 20 EU_transformed', 'Y= - 200 EU_transformed',
       'Y= - 2 000 EU_transformed', 'Y= - 20 000 EU_transformed',
       'Y= - 200 000 EU_transformed']])


# In[169]:


pg.cronbach_alpha(data[[ 
                       'Y=2 EU_transformed', 'Y=20 EU_transformed', 'Y=200 EU_transformed',
       'Y=2 000 EU_transformed', 'Y=20 000 EU_transformed',
       'Y=200 000 EU_transformed', 'Y= - 2 EU_transformed',
       'Y= - 20 EU_transformed', 'Y= - 200 EU_transformed',
       'Y= - 2 000 EU_transformed', 'Y= - 20 000 EU_transformed',
       'Y= - 200 000 EU_transformed']])


# In[170]:


pg.cronbach_alpha(data[[ 
                       ' [X = 1 EU]',
       ' [X = 10 EU]', ' [X = 100 EU]', ' [X = 1 000 EU]', ' [X = 10 000 EU]',
       ' [X = 100 000 EU]', 
                       'Y=2 EU_transformed', 'Y=20 EU_transformed', 'Y=200 EU_transformed',
       'Y=2 000 EU_transformed', 'Y=20 000 EU_transformed',
       'Y=200 000 EU_transformed', 'Y= - 2 EU_transformed',
       'Y= - 20 EU_transformed', 'Y= - 200 EU_transformed',
       'Y= - 2 000 EU_transformed', 'Y= - 20 000 EU_transformed',
       'Y= - 200 000 EU_transformed']])


# In[171]:


pg.cronbach_alpha(data[[' [X = - 1 EU ]', ' [X = - 10 EU]',
       ' [X = - 100 EU]', ' [X = - 1 000 EU]', ' [X = - 10 000 EU]',
       ' [X = - 100 000 EU]', 
                       'Y=2 EU_transformed', 'Y=20 EU_transformed', 'Y=200 EU_transformed',
       'Y=2 000 EU_transformed', 'Y=20 000 EU_transformed',
       'Y=200 000 EU_transformed', 'Y= - 2 EU_transformed',
       'Y= - 20 EU_transformed', 'Y= - 200 EU_transformed',
       'Y= - 2 000 EU_transformed', 'Y= - 20 000 EU_transformed',
       'Y= - 200 000 EU_transformed']])


# In[ ]:




