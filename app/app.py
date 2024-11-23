from flask import Flask, render_template, request, jsonify
import functions as func

app = Flask(__name__)

@app.route('/')
def menu():
    return render_template('menu.html')

@app.route('/eda')
def eda():
    # Run the EDA Functions to generate fresh graphs and outputs
    route_head = func.get_route_data()
    order_head = func.get_order_data()
    route_missing, order_missing = func.eda_missing_values() # 1
    route_num_summary, order_num_summary = func.eda_numerical_summary() # 2

    feature_importance = func.eda_feature_importance(just_df=True) # 8

    return render_template('eda.html',
                            route_head=route_head,
                            order_head=order_head,
                            route_missing=route_missing,
                            order_missing=order_missing,
                            route_num_summary=route_num_summary,
                            order_num_summary=order_num_summary,
                            feature_importance=feature_importance)


@app.route('/managerial_statistics')
def managerial_statistics():
    func.ms_exponential_distribution_analysis()
    func.ms_binomial_distribution_analysis()
    func.ms_negative_binomial_distribution_analysis()
    func.ms_gamma_distribution_analysis
    return render_template('managerial_statistics.html')

@app.route('/managerial_statistics/bayesian_inference', methods=['POST'])
def bayesian_inference():
    try:
        # Extract the selected column from the form data
        column = request.form.get('bayesian-column')
        alpha_prior = int(request.form.get('bayesian-alpha'))
        beta_prior = int(request.form.get('bayesian-beta'))
        
        # Call the function to generate the graph
        func.ms_bayesian_inference_beta(column, alpha_prior, beta_prior)
        
        # Respond with a success message and the path to the updated image
        return jsonify({"success": True, "image_path": "/static/images/ms_1_bayesian_inference_beta.png"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/managerial_statistics/poisson_distribution', methods=['POST'])
def poisson_distribution():
    try:
        # Extract the selected column from the form data
        source = request.form.get('poisson-source')
        destination = request.form.get('poisson-destination')
        min_range = int(request.form.get('poisson-min'))
        max_range = int(request.form.get('poisson-max'))
        
        # Call the function to generate the graph
        func.ms_poisson_distribution_analysis(source, destination, min_range, max_range)
        
        # Respond with a success message and the path to the updated image
        return jsonify({"success": True, "image_path": "/static/images/ms_2_poisson_distribution.png"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
    

@app.route('/optimization_models')
def optimization_models():
    return render_template('optimization_models.html')

@app.route('/optimization_models/transportation_optimization', methods=['POST'])
def transportation_optimization():
    try:
        # Extract the selected column from the form data
        budget = request.form.get('transportation-budget')
        budget = int(budget)
        
        # Call the function to generate the graph and give the result
        result = func.om_transportation_problem_with_budget(budget)
        if result != "Infeasible":
            result = str(round(float(result),5))

        # Respond with a success message and the path to the updated image
        return jsonify({"success": True, "image_path": "/static/images/om_1_shipment_plan.png", "result": result, "budget": budget})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
    
@app.route('/optimization_models/gradient_descent', methods=['POST'])
def gradient_descent():
    try:
        # Extract the selected column from the form data
        learning_rate = request.form.get('gradient-learning-rate')
        learning_rate = float(learning_rate)
        iterations = request.form.get('gradient-iterations')
        iterations = int(iterations)
        
        # Call the function to generate the graph and give the result
        result = func.om_optimize_route_cost(learning_rate=learning_rate, iterations=iterations)
        freight_cost = result['Optimal Freight Cost']
        fuel_cost = result['Optimal Fuel Cost']
        total_cost = result['Minimum Total Cost']

        # Respond with a success message and the path to the updated image
        return jsonify({"success": True,
                        "cost_image_path": "static/images/om_2_cost_function.png",
                        "gradient_image_path":'static/images/om_2_gradient_descent.png', 
                        "freight_cost": freight_cost, 
                        "fuel_cost": fuel_cost, 
                        "total_cost": total_cost})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/optimization_models/knapsack_optimization', methods=['POST'])
def knapsack_optimization():
    try:
        # Extract the selected column from the form data
        capacity = request.form.get('knapsack-capacity') #input taken as a string of numbers seperated by commas
        capacity = list(map(int, capacity.split(',')))
        
        # Call the function to generate the graph and give the result
        result = func.om_balanced_knapsack_optimization(capacity)

        # Respond with a success message and the path to the updated image
        return jsonify({"success": True,
                        "image_path": "static/images/om_3_knapsack.png",
                        "desc_image_path": "static/images/om_3_description.png"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
    
if __name__ == '__main__':
    func.initialize(load_data=False)
    app.run(debug=True)
