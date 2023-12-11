

function global_search_mr(X, y, idx, feature)
    # Convert idx to an array of integers if it's not already
    idx = Int64.(idx)

    # Extract the subset of data for the specified feature and indices
    X_sub = X[idx, feature]
    y_sub = y[idx]

    # Sort the subset of data and get the sorted indices
    sorted_indices = sortperm(X_sub)
    X_sorted = X_sub[sorted_indices]
    y_sorted = y_sub[sorted_indices]

    # Initialize the best loss and split point
    best_loss = Inf
    best_split_point = nothing
    nn = length(y_sorted)
    m = size(y, 2)  # Assuming y is 2D for multi-class classification
    
    # Calculate the cumulative sum for the sorted labels
    Cy = cumsum(y_sorted, dims=1)
    RCy = Cy[end, :]' .- Cy
    Cy_norms = reshape(sum(Cy .* Cy, dims=2), nn)[1:end-1]
    RCy_norms = reshape(sum(RCy .* RCy, dims=2), nn)[1:end-1]

    # Compute losses for each potential split point
    for pos in 1:nn-1
        a = collect(1:pos)
        second_terms = Cy_norms[1:pos] ./ a + RCy_norms[1:pos] ./ reverse(a)
        loss_cur = nn .- maximum(second_terms)
        
        if loss_cur < best_loss
            best_loss = loss_cur
            best_split_point = (X_sorted[pos] + X_sorted[pos + 1]) / 2
        end
    end
    
    return best_loss, best_split_point
end